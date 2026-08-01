# MIT License
#
# Copyright (c) 2026 Gilles-Philippe Morin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Pedigree drawing, the GeneaKit counterpart of GENLIB's `gen.graph()`.

The module is a small pipeline, each stage a pure function of the previous:

    generations -> mating blocks -> ordering -> abscissas -> routing -> ink

`PedigreeLayout` holds the geometry produced by the first five stages and
knows how to draw itself; `graph()` is the thin public wrapper that mirrors
the GENLIB call.

The layout is a layered (Sugiyama-style) drawing specialized for pedigrees.
Individuals are assigned to generational rows, mates are welded into rigid
horizontal blocks, blocks are ordered by iterated barycentre sweeps, and
their abscissas are then fitted by weighted isotonic regression, which is the
exact minimizer of the squared distance to the barycentre targets under the
non-overlap constraints. Unlike kinship2, an individual is not duplicated
unless asked for: an inbreeding loop is drawn as a loop, which is usually
what one wants when looking at a founder population.

The drawing follows the standardized human pedigree nomenclature of the
National Society of Genetic Counselors -- Bennett et al., American Journal of
Human Genetics 56:745 (1995), Journal of Genetic Counseling 17:424 (2008) and
31:1238 (2022) -- so far as a genealogy can. That nomenclature says nothing
about a relationship line spanning two generational rows, since clinical
pedigrees avoid the case by duplicating an individual; where GeneaKit draws
such a line instead, the standard marks are applied to the part of it that
runs straight rather than invented anew.

Everything a pedigree needs beyond the four columns of a genealogy is passed
to `graph()` as an argument, on one principle: what concerns one person takes
that person, what concerns a tie between people takes the tie, and either may
be given alone or mapped to what is known about it.

This module was written with the help of Claude Opus 5.
"""

import math
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ['graph', 'PedigreeLayout']

# Geometry, in layout units (one unit is one individual slot).
_MATE_GAP = 1.0       # Centre-to-centre distance between mates
_BLOCK_GAP = 1.7      # Centre-to-centre distance across families
_ROW_GAP = 1.7        # Smallest distance between generations
_SYMBOL = 0.62        # Side of a square, diameter of a circle
_LABEL = 0.46         # Room left under the symbols for their labels
_CHANNEL = 0.34       # Distance between two horizontal lines of a band
_CLEARANCE = 0.7      # Room left between two lines sharing a channel
_WIRE_GAP = 1.0       # Room left around a descent line crossing a generation
_PIN = 1000.0          # Weight holding such a line under what it hangs from
_DOUBLE = 0.09        # Distance between the two lines of a consanguineous union
_HOP = 0.085          # Radius of the bridge a line makes over one it only crosses
_SNAP = 0.35          # Sibship bar below which a descent is drawn straight
_COURTESY = 0.9       # Cost of drawing a woman to the left of her mate
_ROUTED = 3.0         # Cost of a mating line that has to go round the symbols
_ASIDE = 1.3          # Room a union spanning two rows needs to read as a union
_DETOUR = 1.8         # How far a union may reach for a corridor of its own
_STANDOFF = 0.2       # Room a corridor leaves around the mates it joins
_INSIST = 3           # How firmly such a mate is held to one side of its partner
_MARGIN = 0.26        # Room left between the channels of a band
_LOOSE = 1e-3         # Weight of a block that nothing pulls on
_INCH = 0.42          # Inches to a layout unit, at the size a pedigree reads best
_PAPER = 150.0        # Largest figure, in inches, a drawing may grow to
_WIDE = 60.0          # Figure beyond which a pedigree is too big to draw whole


def graph(gen, **kwargs):
    """Draw a pedigree, as GENLIB's `gen.graph()` does

    Renders a genealogy as a conventional pedigree chart: squares for men,
    circles for women, diamonds for unknown sex, a horizontal line between
    mates, and a sibship bar joining each mating to its children. Generations
    run from the oldest ancestors at the top to the probands at the bottom.

    Args:
        gen (cgeneakit.Pedigree | pd.DataFrame): Genealogy to draw, or a
            pedigree table with columns ['ind', 'father', 'mother', 'sex'].
        pro (list, optional): Probands to keep. Defaults to all individuals
            without children.
        ancestors (list, optional): Ancestors to keep. Defaults to all
            founders. Giving either argument extracts the corresponding
            subpedigree with gen.branching() before drawing.
        labels (dict | pd.Series | list | callable | bool, optional): Text
            printed under each symbol, keyed or aligned by individual ID.
            Defaults to the IDs themselves; pass False to omit the labels.
        col (str | dict | pd.Series | list | callable, optional): Outline and
            label color of each symbol. Defaults to black.
        fill (dict | pd.Series | list | callable, optional): Interior color of
            each symbol. Numeric values are mapped through `cmap`, which makes
            it easy to paint carrier probabilities, inbreeding coefficients or
            genetic contributions onto the pedigree. Defaults to white.
        affected (dict | pd.Series | list | callable, optional): Boolean
            status, or simply the IDs of the affected; they get a solid
            symbol. Superseded by `fill` wherever both are given.
        deceased (dict | pd.Series | list | callable, optional): Boolean
            status, or the IDs of the dead; their symbol is struck through.
        carrier (dict | pd.Series | list | callable, optional): Boolean
            status, or the IDs of the obligate carriers; they get a dot in
            the middle of their symbol.
        proband (dict | pd.Series | list | callable, optional): Boolean
            status, or the IDs of the probands the pedigree was drawn for;
            an arrow points at their symbol from the lower left.
        consanguinity (bool, optional): Draw the union of two related mates
            as a double line, as pedigrees conventionally do. Defaults to
            True.
        pregnancy (dict | pd.Series | list | callable, optional): Boolean
            status, or the IDs of the ongoing pregnancies; their symbol
            carries a P.
        count (dict | pd.Series | list | callable, optional): Number of
            individuals a symbol stands for, written inside it.
        twins (list | dict, optional): Groups of twins, whose lines of descent
            are drawn meeting at a single point of the sibship bar. Give the
            groups alone, [(3, 4), (7, 8, 9)], or map each to its zygosity,
            {(3, 4): 'mz', (7, 8): 'dz'}; an unstated zygosity is drawn as the
            question mark of the convention.
        childless (list | dict, optional): Unions known to have no children,
            which the four columns of a pedigree cannot record. Give the pairs
            alone, [(3, 4)], or map each to the reason, {(3, 4): 'i'}, written
            beside the two bars that close the union.
        cmap (str | Colormap, optional): Colormap for numeric `fill` values.
            Defaults to 'viridis'.
        norm (tuple, optional): (vmin, vmax) for numeric `fill` values.
            Defaults to the observed range.
        colorbar (bool, optional): Draw a colorbar for numeric `fill` values.
            Defaults to True.
        cex (float, optional): Text size multiplier. Defaults to 1.
        symbolsize (float, optional): Symbol size multiplier. Defaults to 1.
        generations (bool, optional): Number the generations in Roman numerals
            down the left margin. Defaults to False.
        align (str, optional): How generations are decided. 'mates' keeps a
            couple on one row, which is what a pedigree normally looks like.
            'descent' puts everyone exactly one row below their deepest
            parent, so that a row number is a true generation of descent and
            the numerals mean what they say; mates of unequal depth are then
            joined across rows. Defaults to 'mates'.
        repeat (int | None, optional): Longest line of descent tolerated, in
            generations, before an individual is drawn a second time among its
            brothers and sisters rather than reached across the drawing. It
            buys a shorter drawing at the price of asking the reader to hold
            two symbols together in mind, which is why it is off by default:
            turn it on for a pedigree whose long lines have become impossible
            to follow. Defaults to None, repeating nobody.
        sweeps (int, optional): Number of ordering passes. Larger values give
            tidier drawings at a linear cost. Defaults to 8.
        ax (matplotlib.axes.Axes, optional): Axes to draw on. A new figure is
            created if omitted.
        figsize (tuple, optional): Figure size in inches when `ax` is omitted.
            Defaults to a size proportional to the drawing.
        title (str, optional): Title added above the pedigree.

    Returns:
        PedigreeLayout: Geometry of the drawing, carrying the same information
            as the list returned by kinship2::plot.pedigree (n, nid, pos, fam,
            spouse, x, y, boxw, boxh), plus the Matplotlib axes in `ax`.

    Examples:
        >>> import geneakit as gen
        >>> import matplotlib.pyplot as plt
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> layout = gen.graph(ped, title='Jicaque pedigree')
        >>> plt.savefig('jicaque.png', dpi=200, bbox_inches='tight')

        ### A subpedigree, painted with a quantitative variable
        >>> from geneakit import genea140
        >>> ped = gen.genealogy(genea140)
        >>> sub = gen.branching(ped, pro=[409033, 408728])
        >>> contributions = gen.gc(sub).loc[409033]
        >>> layout = gen.graph(sub, fill=contributions, labels=False,
        ...                    cmap='magma')

        ### What the four columns cannot hold, given as arguments
        >>> layout = gen.graph(ped,
        ...                    twins={(8, 9): 'mz', (11, 12): 'dz'},
        ...                    childless={(3, 14): 'i'},
        ...                    pregnancy=[10], count={13: 3},
        ...                    deceased=[1, 2], carrier=[5], proband=[11],
        ...                    generations=True)

        ### The same ties, given bare when nothing more is known
        >>> layout = gen.graph(ped, twins=[(8, 9)], childless=[(3, 14)])

    Notes:
        The marks of the standard nomenclature:
        - Squares for men, circles for women, diamonds for unknown sex; solid
          symbols for the affected, a dot for obligate carriers, a stroke
          through the symbol for the dead, an arrow for the probands, a P for
          a pregnancy, and a count inside a symbol standing for several
          people. The middle of a symbol carries one mark at a time, in that
          order of precedence.
        - Twins meet at a single point of the sibship bar, with the
          monozygosity bar laid between their lines of descent rather than
          between the symbols; an unstated zygosity is drawn as a question
          mark in the same place. A union without children is closed by two
          bars, with the reason written beside them if one is given.
        - A consanguineous union is drawn as a double line, and is read from
          the pedigree itself rather than declared. A union that had to be
          joined the long way round is doubled along its longest straight run
          alone, since two strokes carried round a corner read as two lines
          rather than as one relationship.

        How the drawing is kept legible:
        - Every line is straight. No two horizontal lines share an ordinate
          unless they are far apart, and a descent line that has generations
          to cross is given a lane of its own, which the families around it
          make room for. Generations are spaced by the number of lines the
          band between them has to carry.
        - Where a line has no choice but to cross another, it bridges over it,
          so that a crossing is never read as a junction and no one is given
          parents they do not have.
        - An only child is reached by one straight line rather than by a
          sibship bar, but only where that line would still meet the line
          joining its parents; otherwise it is given a proper bar, since a
          drop that begins beside its parents' line looks like an orphan.
        - Men are drawn to the left of their mates wherever that does not
          lengthen the lines around them, and never at the cost of separating
          somebody from a second mate.
        - Lettering takes its size from the symbols, so a pedigree of thirty
          and one of five hundred come out looking alike, while the title and
          the generation numerals are sized from the page and stay readable
          across a wide chart. The figure grows with the pedigree rather than
          being squeezed onto a fixed page; past the size of any real sheet a
          warning suggests reducing the pedigree first.

        On the layout itself:
        - Mates share a row whenever the pedigree allows it. A few pedigrees
          admit no such assignment (a man marrying his son's daughter, say);
          generational depth then takes precedence, a warning is issued, and
          the mating line is drawn across the two rows.
        - A union whose mates end up on different rows is drawn as a step: out
          of the side of the upper mate, down a corridor of its own, and into
          the side of the lower one. Leaving and entering sideways is what
          tells it apart from a line of descent, which always leaves the foot
          of one symbol and enters the head of another. The corridor sits
          between the mates when they leave room for it and in the nearest
          free lane otherwise, clear of the symbols of every row it crosses,
          its own mates included; a union that would have to reach too far for
          one is routed below the symbols instead.
        - Individuals are not duplicated unless `repeat` asks for it, so an
          inbreeding loop is drawn as a loop. A repeat is drawn in broken
          strokes and lettered like the first showing, and `repeats` on the
          returned layout says which symbol stands for whom.
        - The GENLIB arguments `packed`, `align` and `width` steer the
          kinship2 layout engine and have no counterpart here; `sweeps` is the
          only knob of this one.
        - Laying out a pedigree of a few hundred individuals takes well under
          a second, and the whole of genea140 about five, but a chart of more
          than a few hundred people is rarely legible: reach for
          gen.branching() or gen.lineages() first.

    See Also:
        gen.branching: Extract the subpedigree of given probands and ancestors
        gen.lineages: Extract maternal or paternal lines
        gen.genout: Export the same pedigree as a DataFrame
    """
    frame = _pedigree_frame(gen, kwargs.pop('pro', None),
                            kwargs.pop('ancestors', None))
    layout = PedigreeLayout.from_frame(
        frame, twins=kwargs.pop('twins', None),
        childless=kwargs.pop('childless', None),
        align=kwargs.pop('align', 'mates'),
        repeat=kwargs.pop('repeat', None),
        sweeps=kwargs.pop('sweeps', 8))
    layout.draw(**kwargs)
    return layout


@dataclass
class Mating:
    """Geometry of one mating and of the sibship hanging from it

    Attributes:
        father (int): Paternal ID, 0 if unknown.
        mother (int): Maternal ID, 0 if unknown.
        mates (tuple): IDs of the known parents, from left to right.
        children (list): IDs of the children.
        consanguineous (bool): Whether the mates share an ancestor, in which
            case the union is drawn as a double line.
        twins (list): (group, zygosity) pairs among the children, whose lines
            of descent meet at a single point on the sibship bar.
        childless (str | bool | None): Set when the union is known to have no
            children, to the reason if one was given.
        line (float | None): Ordinate of the mating line when two mates of one
            row are too far apart to be joined directly, None otherwise.
        corridor (float | None): Abscissa at which a union spanning two rows
            steps down from one row to the other, None when it does not.
        origin (tuple): Point the descent line leaves the mating from.
        bar (float): Ordinate of the sibship bar.
    """

    father: int
    mother: int
    mates: tuple
    children: list
    twins: list
    childless: object
    consanguineous: bool
    line: float
    corridor: float
    origin: tuple
    bar: float


@dataclass
class PedigreeLayout:
    """Geometry of a pedigree drawing

    Attributes:
        ind (np.ndarray): Individual IDs, one entry per individual.
        sex (np.ndarray): Sex codes (1=male, 2=female, 0=unknown).
        x (np.ndarray): Abscissa of each individual, in layout units.
        y (np.ndarray): Ordinate of each individual, in layout units.
        row (np.ndarray): Generation of each individual, 0 at the top.
        slot (np.ndarray): Rank of each individual within its generation.
        levels (np.ndarray): Ordinate of each generation.
        matings (list): Geometry of every mating: see `Mating`.
        n (np.ndarray): Number of individuals per generation.
        nid (np.ndarray): Individual IDs, one padded row per generation.
        pos (np.ndarray): Abscissas, laid out like `nid`.
        fam (np.ndarray): One-based rank, in the row above, of the father of
            each individual; 0 when the parents are elsewhere or absent.
        spouse (np.ndarray): 1 where an individual mates with the one on its
            right.
        repeats (dict): Maps each repeated symbol to the individual it stands
            for; empty when nobody had to be drawn twice.
        boxw (float): Symbol width.
        boxh (float): Symbol height.
        ax (matplotlib.axes.Axes): Axes the pedigree was drawn on, if any.

    Notes:
        - `n`, `nid`, `pos`, `fam` and `spouse` reproduce the `plist` returned
          by kinship2::plot.pedigree, so that code ported from R keeps working;
          the per-individual arrays are usually handier in Python.
        - The layout is deterministic: a given pedigree always yields the same
          drawing.
        - Lettering is sized from the symbols rather than fixed in points, so
          a pedigree of thirty and one of five hundred come out looking alike.

    Examples:
        >>> import geneakit as gen
        >>> from geneakit import geneaJi
        >>> ped = gen.genealogy(geneaJi)
        >>> layout = gen.graph(ped)
        >>> layout.coordinates.head(3)
              x    y  row
        ind
        17  0.0  0.0    0
        19  4.6  0.0    0
        25  9.2  0.0    0
    """

    ind: np.ndarray
    sex: np.ndarray
    x: np.ndarray
    y: np.ndarray
    row: np.ndarray
    slot: np.ndarray
    levels: np.ndarray
    matings: list
    n: np.ndarray
    nid: np.ndarray
    pos: np.ndarray
    fam: np.ndarray
    spouse: np.ndarray
    repeats: dict = field(default_factory=dict)
    boxw: float = _SYMBOL
    boxh: float = _SYMBOL
    ax: object = field(default=None, repr=False)

    @classmethod
    def from_frame(cls, frame, twins=None, childless=None, align='mates',
                   repeat=None, sweeps=8):
        """Compute the layout of a pedigree table, without drawing it

        Args:
            frame (pd.DataFrame): Table with columns ['ind', 'father',
                'mother', 'sex'], as returned by gen.genout().
            twins (list | dict, optional): Groups of twins; see gen.graph().
            childless (list | dict, optional): Unions without children.
            align (str, optional): 'mates' or 'descent'; see gen.graph().
            repeat (int | None, optional): Longest descent line tolerated
                before an individual is drawn twice. Defaults to None.
            sweeps (int, optional): Number of ordering passes. Defaults to 8.

        Returns:
            PedigreeLayout: Coordinates of every individual and mating.

        Examples:
            >>> import geneakit as gen
            >>> from geneakit import geneaJi
            >>> ped = gen.genealogy(geneaJi)
            >>> layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
            >>> layout.n
            array([...])
        """
        frame = _columns(frame)
        blood = {int(one): (int(father), int(mother)) for one, father, mother
                 in zip(frame['ind'], frame['father'], frame['mother'])}
        repeats = {}
        if repeat is not None:
            frame, repeats = _repeat(frame, repeat, align)
        pedigree = _Pedigree(frame, twins, childless, align, blood)
        rows, x = _arrange(pedigree, sweeps)
        return cls(repeats=repeats, **pedigree.geometry(rows, x))

    def draw(self, **kwargs):
        """Render the layout with Matplotlib

        Args:
            **kwargs (Any): Drawing options; see gen.graph() for the full
                list.

        Returns:
            matplotlib.axes.Axes: The axes the pedigree was drawn on.
        """
        self.ax = _draw(self, **kwargs)
        return self.ax

    @property
    def families(self):
        """Matings as (father, mother, children) tuples of individual IDs"""
        return [(mating.father, mating.mother, mating.children)
                for mating in self.matings]

    @property
    def coordinates(self):
        """Individual coordinates, as a DataFrame indexed by ID"""
        return pd.DataFrame({'x': self.x, 'y': self.y, 'row': self.row},
                            index=pd.Index(self.ind, name='ind'))


# -----------------------------------------------------------------------------
# Pedigree bookkeeping
# -----------------------------------------------------------------------------


class _Pedigree:
    """Index-based view of a pedigree table, with its matings and generations

    Individuals are referred to by their position in the table; an unknown
    parent is -1, and every mating is a (parents, children) pair of index
    tuples, listed in order of first appearance so that the drawing does not
    depend on hashing.
    """

    def __init__(self, frame, twins=None, childless=None, align='mates',
                 blood=None):
        frame = _columns(frame)
        self.ids = frame['ind'].to_numpy(dtype=np.int64)
        if len(np.unique(self.ids)) != self.ids.size:
            raise ValueError('The pedigree has duplicate individual IDs')
        self.sex = frame['sex'].to_numpy(dtype=np.int64)
        self.size = self.ids.size
        rank = {identifier: index for index, identifier in enumerate(self.ids)}
        self.father, self.mother = (
            np.array([rank.get(parent, -1) for parent in frame[column]],
                     dtype=np.int64) for column in ('father', 'mother'))
        self.align = align
        self.blood = tuple(
            np.array([rank.get((blood or {}).get(identifier, (0, 0))[side], -1)
                      for identifier in self.ids], dtype=np.int64)
            for side in (0, 1)) if blood else (self.father, self.mother)
        self.matings, self.as_parent, self.as_child = self._matings()
        self.twins = [(members, zygosity) for members, zygosity in
                      ((tuple(rank[member] for member in group if member in rank),
                        note) for group, note in _groups(twins))
                      if len(members) > 1]
        self.childless = {}
        for group, note in _groups(childless):
            union = tuple(rank[mate] for mate in group if mate in rank)
            if union:
                self.childless[len(self.matings)] = note
                self.matings.append((union, []))
        self.generation = self._generations()
        self.depth = int(self.generation.max()) + 1 if self.size else 0
        self.wires, self.lane, self.level = self._wires()
        self.nodes = self.level.size

    def _matings(self):
        """Group children by parental couple and index the couples both ways"""
        matings, where = [], {}
        as_parent = [[] for _ in range(self.size)]
        as_child = np.full(self.size, -1, dtype=np.int64)
        for child, parents in enumerate(zip(self.father, self.mother)):
            if max(parents) < 0:
                continue
            if parents not in where:
                where[parents] = len(matings)
                matings.append((tuple(p for p in parents if p >= 0), []))
                for parent in matings[-1][0]:
                    as_parent[parent].append(where[parents])
            matings[where[parents]][1].append(child)
            as_child[child] = where[parents]
        return matings, as_parent, as_child

    def _generations(self):
        """Assign each individual to a generational row

        Two requirements compete: children belong strictly below both of their
        parents, and mates belong on the same row. Welding each couple into a
        single node turns them into one, since the rows are then simply the
        longest paths of the contracted descent graph. Under `align='descent'`
        only the couples with a founder among them are welded, since a founder
        has no depth of its own and belongs beside its mate; everyone whose
        ancestry is known then falls exactly one row below their deepest
        parent, so that a row number is a true generation of descent. A few pedigrees admit
        no such assignment at all, the couples of the contracted graph forming
        a cycle -- a man marrying his son's daughter, say. The offending
        couples are unwelded, one round of strongly connected components at a
        time, until descent alone decides; those mates end up one row apart.

        Returns:
            np.ndarray: Row of each individual, 0 for the oldest generation.

        Raises:
            ValueError: If an individual is one of its own ancestors.
        """
        couples = [parents for parents, _ in self.matings if len(parents) == 2
                   and (self.align != 'descent' or any(
                       max(self.father[mate], self.mother[mate]) < 0
                       for mate in parents))]
        while True:
            welded = self._weld(couples)
            edges = self._edges(welded)
            tangled = self._tangled(welded, edges)
            if not tangled:
                break
            kept = [parents for parents in couples
                    if welded[parents[0]] not in tangled]
            if len(kept) == len(couples):
                raise ValueError('The pedigree contains a cycle of descent')
            couples = kept
            warnings.warn('Some mates cannot share a generation; their mating '
                          'lines span two rows', stacklevel=4)
        levels = self._levels(welded, edges)
        return np.array([levels[node] for node in welded], dtype=np.int64)

    def _weld(self, couples):
        """Merge mates into shared nodes, and label everyone with their node"""
        node = list(range(self.size))

        def root(individual):
            while node[individual] != individual:
                node[individual] = node[node[individual]]
                individual = node[individual]
            return individual

        for mates in couples:
            left, right = root(mates[0]), root(mates[1])
            if left != right:
                node[left] = right
        return [root(individual) for individual in range(self.size)]

    def _edges(self, welded):
        """Descent edges between welded nodes, as (parent, child) pairs"""
        return {(welded[parent], welded[child])
                for child in range(self.size)
                for parent in (self.father[child], self.mother[child])
                if parent >= 0}

    @staticmethod
    def _tangled(welded, edges):
        """Welded nodes caught in a cycle of descent, if any

        Returns:
            set: Nodes lying on a cycle, empty when the contracted graph is
            the directed acyclic graph a pedigree drawing needs.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        if not edges:
            return set()
        tangled = {parent for parent, child in edges if parent == child}
        nodes = sorted(set(welded))
        rank = {node: index for index, node in enumerate(nodes)}
        arcs = np.array([(rank[parent], rank[child]) for parent, child in edges
                         if parent != child] or [(0, 0)]).T
        _, labels = connected_components(
            coo_matrix((np.ones(arcs.shape[1]), arcs),
                       shape=(len(nodes), len(nodes))),
            directed=True, connection='strong')
        sizes = np.bincount(labels)
        return tangled | {node for node, label in zip(nodes, labels)
                          if sizes[label] > 1}

    @staticmethod
    def _levels(welded, edges):
        """Longest path down to each welded node, the row it will be drawn on"""
        children, pending = {}, dict.fromkeys(welded, 0)
        for parent, child in edges:
            children.setdefault(parent, []).append(child)
            pending[child] += 1
        levels = dict.fromkeys(pending, 0)
        order = [node for node, count in pending.items() if not count]
        for node in order:
            for child in children.get(node, ()):
                levels[child] = max(levels[child], levels[node] + 1)
                pending[child] -= 1
                if not pending[child]:
                    order.append(child)
        return levels

    def _wires(self):
        """Reserve a lane for every descent line that has generations to cross

        Most descent lines drop from one generation straight into the next and
        need no room of their own. A child pulled several rows down by a deeper
        mate, though, is reached by a line that has to make its way past whole
        generations. Such a line is given a node of its own in every row it
        crosses, which then takes part in the ordering and the spacing exactly
        as an individual does: the drawing opens a lane for it instead of
        letting it fend for itself among the symbols.

        Returns:
            tuple: (wires, lane, level), the nodes standing in for each
            crossing line, what each of those nodes hangs from, and the row
            every node of the drawing belongs to.
        """
        wires, lane, rows, node = [], {}, [], self.size
        for index, (parents, children) in enumerate(self.matings):
            if not children:
                continue
            top = min(self.generation[child] for child in children)
            below = max(self.generation[parent] for parent in parents) + 1
            crossings = [(range(below, top), ('mating', index))]
            crossings += [(range(top, self.generation[child]), ('child', child))
                          for child in children]
            for levels, anchor in crossings:
                if not len(levels):
                    continue
                chain = list(range(node, node + len(levels)))
                node += len(chain)
                rows.extend(levels)
                lane.update(dict.fromkeys(chain, anchor))
                wires.append((chain, anchor))
        return wires, lane, np.concatenate(
            [self.generation, np.array(rows, dtype=np.int64)])

    def related(self, mates, budget=200000):
        """Whether two mates share an ancestor, or one descends from the other

        Ancestors are gathered from both mates a generation at a time and the
        two collections compared after each step, so that a union between close
        relatives -- the common case in a founder population -- is settled in a
        few steps rather than by walking two whole ancestries.

        Returns:
            bool: True if the union is consanguineous within this pedigree.
            Unrelated mates of very deep ancestry exhaust a fixed budget of
            work first, and are reported as unrelated.
        """
        if len(mates) < 2:
            return False
        seen = [{mate} for mate in mates]
        rising = [{mate} for mate in mates]
        while any(rising) and sum(map(len, seen)) < budget:
            for side, front in enumerate(rising):
                rising[side] = {
                    parent for individual in front
                    for parent in (self.blood[0][individual],
                                   self.blood[1][individual])
                    if parent >= 0 and parent not in seen[side]}
                seen[side] |= rising[side]
            if seen[0] & seen[1]:
                return True
        return False

    def anchor(self, lane, x):
        """Abscissa a crossing line is meant to keep to"""
        kind, target = lane
        return self.parents_centre(target, x) if kind == 'mating' else x[target]

    def parents_centre(self, mating, x):
        """Abscissa the children of a mating hang from"""
        parents = self.matings[mating][0]
        return (x[parents[0]] + x[parents[-1]]) / 2

    def children_centre(self, mating, x):
        """Abscissa the parents of a mating are drawn towards"""
        children = self.matings[mating][1]
        return sum(x[child] for child in children) / len(children)

    def geometry(self, rows, x):
        """Assemble the public, kinship2-compatible geometry"""
        rows = [[node for node in row if node < self.size] for row in rows]
        slot = np.zeros(self.size, dtype=np.int64)
        for row in rows:
            for rank, individual in enumerate(row):
                slot[individual] = rank
        width = max((len(row) for row in rows), default=0)
        nid, pos, fam, spouse = (np.zeros((len(rows), width), dtype=dtype)
                                 for dtype in (np.int64, float, np.int64,
                                               np.int64))
        for index, row in enumerate(rows):
            nid[index, :len(row)] = self.ids[row]
            pos[index, :len(row)] = x[row]
        for individual in range(self.size):
            parent = max(self.father[individual], self.mother[individual])
            if parent >= 0 and \
                    self.generation[parent] == self.generation[individual] - 1:
                fam[self.generation[individual], slot[individual]] = \
                    slot[parent] + 1
        for parents, _ in self.matings:
            if len(parents) == 2 and \
                    self.generation[parents[0]] == self.generation[parents[1]]:
                left, right = sorted(slot[parent] for parent in parents)
                if right == left + 1:
                    spouse[self.generation[parents[0]], left] = 1
        levels, matings = _route(self, slot, x)
        return dict(ind=self.ids, sex=self.sex, x=np.asarray(x)[:self.size],
                    y=np.array([levels[row] for row in self.generation]),
                    row=self.generation, slot=slot, levels=levels,
                    matings=matings,
                    n=np.array([len(row) for row in rows], dtype=np.int64),
                    nid=nid, pos=pos, fam=fam, spouse=spouse)


def _columns(frame):
    """The four columns of a pedigree, whatever they were called"""
    columns = ['ind', 'father', 'mother', 'sex']
    return frame.loc[:, columns] if set(columns) <= set(frame.columns) \
        else frame.iloc[:, :4].set_axis(columns, axis=1)


def _repeat(frame, span, align):
    """Repeat the individuals whose line of descent would cross too many rows

    An individual married far below its parents drags a line across every
    generation in between, which is the worst thing that can happen to a
    pedigree drawing. Rather than draw that line, the individual is entered
    twice and the line is cut. Which of the two ties is cut depends on the
    alignment: where mates share a row it is the tie to the parents, so the
    repeat joins its brothers and sisters where it belongs by birth; where rows
    are true generations it is the tie to the mate, so the repeat goes down to
    the marriage. Either way the repeat is drawn in broken strokes under the
    same label, which is how a reader knows the two are one person.

    Returns:
        tuple: (frame, repeats), the pedigree with its repeats added and a
        mapping from each repeat to the individual it stands for.
    """
    pedigree = _Pedigree(frame, align=align)
    father = frame['father'].to_numpy().copy()
    mother = frame['mother'].to_numpy().copy()
    spare = int(frame['ind'].max()) + 1 if pedigree.size else 1
    added, repeats = [], {}
    for parents, children in pedigree.matings if align == 'descent' else ():
        if len(parents) != 2 or abs(pedigree.generation[parents[0]]
                                    - pedigree.generation[parents[1]]) <= span:
            continue
        early = min(parents, key=lambda parent: pedigree.generation[parent])
        repeats[spare] = int(pedigree.ids[early])
        added.append({'ind': spare, 'father': 0, 'mother': 0,
                      'sex': int(pedigree.sex[early])})
        for child in children:
            column = father if pedigree.father[child] == early else mother
            column[child] = spare
        spare += 1
    for index in range(pedigree.size) if align != 'descent' else ():
        parents = [parent for parent in (pedigree.father[index],
                                         pedigree.mother[index]) if parent >= 0]
        if not parents or pedigree.generation[index] - max(
                pedigree.generation[parent] for parent in parents) <= span:
            continue
        repeats[spare] = int(pedigree.ids[index])
        added.append({'ind': spare, 'father': int(father[index]),
                      'mother': int(mother[index]),
                      'sex': int(pedigree.sex[index])})
        father[index] = mother[index] = 0
        spare += 1
    if not repeats:
        return frame, repeats
    frame = frame.copy()
    frame['father'], frame['mother'] = father, mother
    return pd.concat([frame, pd.DataFrame(added)], ignore_index=True), repeats


def _pedigree_frame(gen, pro, ancestors):
    """Coerce a genealogy or a table into a pedigree DataFrame"""
    if isinstance(gen, pd.DataFrame):
        if pro is None and ancestors is None:
            return gen
        from .create import genealogy
        gen = genealogy(gen)
    if pro is not None or ancestors is not None:
        from .extract import branching
        given = (('pro', pro), ('ancestors', ancestors))
        gen = branching(gen, **{key: value for key, value in given
                                if value is not None})
    from .output import genout
    return genout(gen)


# -----------------------------------------------------------------------------
# Ordering and placement
# -----------------------------------------------------------------------------


def _arrange(pedigree, sweeps):
    """Order every generation and place every individual

    Blocks of mates are swept top-down against the position of their parents
    and bottom-up against that of their children, each sweep reordering the
    blocks of a row and refitting their abscissas. Sweeps are cheap and not
    guaranteed to improve matters, so the shortest arrangement seen is kept.

    Returns:
        tuple: (rows, x), the individuals of each generation in drawing order
        and the abscissa of every individual.
    """
    blocks = _blocks(pedigree)
    offset = _offsets(blocks, pedigree.nodes)
    x = [0.0] * pedigree.nodes
    for row in blocks:
        _fit(pedigree, row, offset, x)
    best = (_ink(pedigree, x), _rows(blocks), list(x))
    stale = 0
    for upward in [False, True, None] * max(sweeps, 0):
        levels = range(pedigree.depth)
        for level in reversed(levels) if upward else levels:
            row = blocks[level]
            pulls = _pulls(pedigree, row, x, offset, upward)
            ranks = sorted(range(len(row)),
                           key=lambda rank: (pulls[rank][0], x[row[rank][0]]))
            blocks[level] = [row[rank] for rank in ranks]
            _fit(pedigree, blocks[level], offset, x,
                 [pulls[rank] for rank in ranks])
        _orient(pedigree, blocks, offset, x)
        ink = _ink(pedigree, x)
        stale = 0 if ink < best[0] - 1e-9 else stale + 1
        if not stale:
            best = (ink, _rows(blocks), list(x))
        elif stale == 3:
            break
    rows, x = best[1], best[2]
    _convention(pedigree, rows, x)
    return rows, np.array(x)


def _convention(pedigree, rows, x):
    """Put the man to the left of the woman in each couple, where it is cheap

    Pedigrees are conventionally drawn that way. Exchanging two mates who are
    already side by side moves nothing else in the drawing -- they simply take
    each other's place -- so the convention can be settled once the layout is
    final, and each exchange kept only where it pays for itself against the
    lines it lengthens. Somebody married twice is left alone: they sit between
    their two mates, and there is no exchanging them without tearing one of the
    two unions apart.
    """
    slot = {node: rank for row in rows for rank, node in enumerate(row)}
    partners = {}
    for parents, _ in pedigree.matings:
        for mate in parents:
            partners.setdefault(mate, set()).update(set(parents) - {mate})
    for parents, _ in pedigree.matings:
        if len(parents) != 2:
            continue
        left, right = sorted(parents, key=lambda parent: x[parent])
        if pedigree.sex[left] != 2 or pedigree.sex[right] != 1 or \
                abs(slot[left] - slot[right]) != 1 or \
                pedigree.generation[left] != pedigree.generation[right] or \
                len(partners[left]) > 1 or len(partners[right]) > 1:
            continue
        before = _ink(pedigree, x)
        x[left], x[right] = x[right], x[left]
        if _ink(pedigree, x) > before:
            x[left], x[right] = x[right], x[left]
            continue
        row = rows[pedigree.generation[left]]
        row[slot[left]], row[slot[right]] = right, left
        slot[left], slot[right] = slot[right], slot[left]


def _blocks(pedigree):
    """Weld mates and twins into rigid blocks, one list of blocks per generation

    Two individuals of the same row who have children together, or who were
    born together, are drawn side by side, so the connected components of that
    graph restricted to a row travel as single units. Each component is
    linearized by a depth-first walk from a least-connected member, which
    leaves a twice-married individual between the two mates.

    Returns:
        list: One list of blocks per generation, each block a list of indices.
    """
    beside = _adjacent(pedigree)
    rows = [[] for _ in range(pedigree.depth)]
    seen = np.zeros(pedigree.size, dtype=bool)
    for start in sorted(beside, key=lambda one: (len(beside[one]), one)):
        if not seen[start]:
            rows[pedigree.generation[start]].append(
                _walk(beside, seen, start))
    for individual in np.flatnonzero(~seen):
        rows[pedigree.generation[individual]].append([individual])
    for chain, _ in pedigree.wires:
        for node in chain:
            rows[pedigree.level[node]].append([node])
    return [sorted(row, key=min) for row in rows]


def _adjacent(pedigree):
    """Who must be drawn beside whom: mates of one row, and twins

    Returns:
        dict: Neighbours of every individual that has any.
    """
    ties = [parents for parents, _ in pedigree.matings if len(parents) == 2]
    ties += [pair for members, _ in pedigree.twins
             for pair in zip(members, members[1:])]
    beside = {}
    for left, right in ties:
        if pedigree.generation[left] == pedigree.generation[right]:
            beside.setdefault(left, []).append(right)
            beside.setdefault(right, []).append(left)
    return beside


def _walk(beside, seen, start):
    """Linearize one component, best-connected neighbour first

    Returns:
        list: The members of the component, in the order they are drawn.
    """
    block, stack, seen[start] = [], [start], True
    while stack:
        member = stack.pop()
        block.append(member)
        for other in sorted(beside[member],
                            key=lambda other: -len(beside[other])):
            if not seen[other]:
                seen[other] = True
                stack.append(other)
    return block


def _offsets(blocks, size):
    """Abscissa of each individual relative to the start of its block"""
    offset = [0.0] * size
    for row in blocks:
        for block in row:
            for rank, individual in enumerate(block):
                offset[individual] = rank * _MATE_GAP
    return offset


def _rows(blocks):
    """Concatenate the blocks of each generation into a plain ordering"""
    return [[individual for block in row for individual in block]
            for row in blocks]


def _pulls(pedigree, row, x, offset, upward):
    """Where each block of a row is drawn, and how strongly

    A block is attracted by the matings its members take part in: downwards to
    the midpoint of their parents, upwards to the barycentre of their children.
    The weight is the number of such attachments, so a well-connected block
    outweighs a loosely connected one.

    Args:
        upward (bool | None): Pull towards children if True, towards parents
            if False, towards both if None.

    Returns:
        list: One (target, weight) pair per block, aligned with `row`.
    """
    pulls = []
    for block in row:
        if block[0] >= pedigree.size:
            pulls.append((pedigree.anchor(pedigree.lane[block[0]], x), _PIN))
            continue
        centres = []
        for member in block:
            for mating in pedigree.as_parent[member]:
                mates = pedigree.matings[mating][0]
                if len(mates) != 2:
                    continue
                other = mates[0] if mates[1] == member else mates[1]
                if pedigree.generation[other] != pedigree.generation[member]:
                    centres += [x[other] + _ASIDE * (
                        1 if x[member] >= x[other] else -1)
                        - offset[member]] * _INSIST
            if upward is not True and pedigree.as_child[member] >= 0:
                centres.append(
                    pedigree.parents_centre(pedigree.as_child[member], x)
                    - offset[member])
            if upward is not False:
                centres.extend(pedigree.children_centre(mating, x)
                               - offset[member]
                               for mating in pedigree.as_parent[member])
        pulls.append((sum(centres) / len(centres) if centres else x[block[0]],
                      len(centres)))
    return pulls


def _fit(pedigree, row, offset, x, pulls=None):
    """Place every block of a row as close to its target as room allows

    Minimizing the weighted squared distance to the targets subject to a
    minimum gap between consecutive blocks looks like a quadratic program, but
    shifting each block left by the gaps that precede it turns the constraints
    into plain monotonicity: the exact solution is then a weighted isotonic
    regression, computed by pooling adjacent violators in linear time. The
    positions are absolute, which is what lets a reserved lane hold its place
    while the families around it give way.
    """
    if not row:
        return
    shift, cumulated, previous = [], 0.0, None
    for block in row:
        if previous is not None:
            cumulated += (len(previous) - 1) * _MATE_GAP + _clearance(
                pedigree, previous, block)
        shift.append(cumulated)
        previous = block
    if pulls is None:
        targets, weights = shift, [_LOOSE] * len(row)
    else:
        targets = [target for target, _ in pulls]
        weights = [weight or _LOOSE for _, weight in pulls]
    fitted = _isotonic([target - start for target, start in zip(targets, shift)],
                       weights)
    for block, value, start in zip(row, fitted, shift):
        for member in block:
            x[member] = value + start + offset[member]


def _clearance(pedigree, left, right):
    """Room two neighboring blocks of a row need between them

    A lane carrying a descent line asks for less than a family does, but for
    enough that the line it stands for is never mistaken for a line of the
    families on either side.
    """
    lanes = int(left[0] >= pedigree.size) + int(right[0] >= pedigree.size)
    return (_BLOCK_GAP, _WIRE_GAP, _WIRE_GAP * 0.7)[lanes]


def _isotonic(values, weights):
    """Closest non-decreasing sequence to `values` (pool adjacent violators)"""
    pools = []
    for value, weight in zip(values, weights):
        pools.append([weight, weight * value, 1])
        while len(pools) > 1 and \
                pools[-2][1] * pools[-1][0] >= pools[-1][1] * pools[-2][0]:
            weight, total, count = pools.pop()
            pools[-1][0] += weight
            pools[-1][1] += total
            pools[-1][2] += count
    return [total / weight for weight, total, count in pools
            for _ in range(count)]


def _orient(pedigree, blocks, offset, x):
    """Mirror the blocks that read better backwards, fathers first on ties"""
    for row in blocks:
        for block in row:
            if len(block) < 2:
                continue
            start = x[block[0]]
            centres = [_centre(pedigree, member, x) for member in block]
            span = (len(block) - 1) * _MATE_GAP
            forward = sum(abs(centre - start - rank * _MATE_GAP)
                          for rank, centre in enumerate(centres)
                          if centre is not None)
            backward = sum(abs(centre - start - span + rank * _MATE_GAP)
                           for rank, centre in enumerate(centres)
                           if centre is not None)
            if backward < forward - 1e-9 or (
                    abs(backward - forward) <= 1e-9
                    and pedigree.sex[block[0]] == 2
                    and pedigree.sex[block[-1]] == 1):
                block.reverse()
                for rank, member in enumerate(block):
                    offset[member] = rank * _MATE_GAP
                    x[member] = start + offset[member]


def _centre(pedigree, individual, x):
    """Abscissa an individual is drawn towards, or None if nothing pulls on it"""
    centres = [pedigree.children_centre(mating, x)
               for mating in pedigree.as_parent[individual]]
    if pedigree.as_child[individual] >= 0:
        centres.append(pedigree.parents_centre(pedigree.as_child[individual], x))
    return sum(centres) / len(centres) if centres else None


def _ink(pedigree, x):
    """What the drawing will cost: the length of its lines, plus its breaches

    Line length is the whole of it save for one term, the price of a couple
    drawn against the convention, which lets the two be traded off.
    """
    total = 0.0
    for parents, children in pedigree.matings:
        centre = (x[parents[0]] + x[parents[-1]]) / 2
        total += abs(x[parents[0]] - x[parents[-1]])
        if len(parents) == 2:
            left, right = sorted(parents, key=lambda parent: x[parent])
            total += _COURTESY * (pedigree.sex[left] == 2
                                  and pedigree.sex[right] == 1)
            if pedigree.generation[left] != pedigree.generation[right]:
                total += _ROUTED * max(_ASIDE - (x[right] - x[left]), 0.0)
            elif x[right] - x[left] > _MATE_GAP * 1.5:
                total += _ROUTED
        if not children:
            continue
        offspring = [x[child] for child in children]
        total += abs(centre - sum(offspring) / len(offspring))
        total += max(max(offspring), centre) - min(min(offspring), centre)
    for chain, lane in pedigree.wires:
        centre = pedigree.anchor(lane, x)
        total += sum(abs(x[node] - centre) for node in chain)
    return total


# -----------------------------------------------------------------------------
# Routing: horizontal lines are packed into channels, rows are spaced to fit
# -----------------------------------------------------------------------------


def _route(pedigree, slot, x):
    """Give every mating a sibship bar and every generation an ordinate

    Horizontal lines are what a reader follows to tell who descends from whom,
    so two of them may share an ordinate only if they stay well apart. The
    lines of a band -- the mating lines too wide to be drawn between neighbors,
    then the sibship bars -- are therefore packed into channels by greedy
    interval colouring, which needs exactly as many channels as the largest
    set of mutually overlapping lines. Generations are then spaced by the room
    the channels of the band between them turned out to need, so that a
    crowded band opens up and a simple one stays tight.

    Returns:
        tuple: (levels, matings), the ordinate of each generation and the
        drawing geometry of every mating.
    """
    rows, plans, lines, bars = pedigree.generation, [], {}, {}
    for index, (parents, children) in enumerate(pedigree.matings):
        mates = sorted(parents, key=lambda parent: x[parent])
        left, right = mates[0], mates[-1]
        astride = len(mates) == 2 and rows[left] != rows[right]
        straight = len(mates) == 1 or astride or (
            abs(int(slot[left]) - int(slot[right])) == 1)
        band = max(rows[left], rows[right])
        reach = [x[child] for child in children] + [(x[left] + x[right]) / 2]
        plan = {'mates': mates, 'children': children, 'index': index,
                'centre': (x[left] + x[right]) / 2,
                'band': band,
                'above': min((rows[child] for child in children),
                             default=0) - 1 if children else None,
                'extent': (min(reach), max(reach)), 'straight': straight,
                'astride': astride}
        plans.append(plan)
        if children:
            bars.setdefault(plan['above'], []).append(plan)
    corridors, astride = [], [plan for plan in plans if plan['astride']]
    for plan in sorted(astride, key=lambda plan: _apart(plan['mates'], x)):
        plan['corridor'] = _corridor(pedigree, plan['mates'], x, corridors)
        middle = (x[plan['mates'][0]] + x[plan['mates'][-1]]) / 2
        if abs(plan['corridor'] - middle) > _DETOUR:
            plan['astride'] = plan['straight'] = False
        else:
            corridors.append(plan['corridor'])
    for plan in plans:
        if not plan['straight']:
            lines.setdefault(plan['band'], []).append(plan)
    heights = []
    for band in range(pedigree.depth):
        crossing = lines.get(band, ())
        hanging = bars.get(band, ())
        for plan, channel in zip(crossing, _channels(
                [(x[plan['mates'][0]], x[plan['mates'][-1]])
                 for plan in crossing])):
            plan['line'] = channel
        for plan, channel in zip(hanging, _channels(
                [plan['extent'] for plan in hanging])):
            plan['bar'] = channel
        channels = sum(max((plan[key] for plan in group), default=-1) + 1
                       for key, group in (('line', crossing), ('bar', hanging)))
        heights.append(max(_ROW_GAP, _SYMBOL + _LABEL + _MARGIN
                           + _CHANNEL * channels))
    levels = -np.concatenate(([0.0], np.cumsum(heights[:-1]))) \
        if heights else np.zeros(0)
    return levels, [_mating(pedigree, plan, levels, x) for plan in plans]


def _channels(intervals):
    """Pack overlapping lines into as few channels as they need

    Sweeping the lines from left to right and dropping each into the first
    channel it clears is the classic greedy colouring of an interval graph:
    it is optimal, and it keeps neighboring lines on neighboring channels.

    Returns:
        list: Channel of each line, numbered from the parents outwards.
    """
    ends, channels = [], [0] * len(intervals)
    for index in sorted(range(len(intervals)), key=lambda i: intervals[i][0]):
        left, right = intervals[index]
        for channel, end in enumerate(ends):
            if end + _CLEARANCE <= left:
                ends[channel], channels[index] = right, channel
                break
        else:
            ends.append(right)
            channels[index] = len(ends) - 1
    return channels


def _mating(pedigree, plan, levels, x):
    """Turn a routing plan into the drawing geometry of one mating"""
    mates, row = plan['mates'], pedigree.generation[plan['mates'][0]]
    line = None if plan['straight'] else (
        levels[plan['band']] - _SYMBOL / 2 - _LABEL
        - (plan['line'] + 0.5) * _CHANNEL)
    corridor = plan['corridor'] if plan['astride'] else None
    if len(mates) == 1:
        origin = (x[mates[0]], levels[row] - _SYMBOL / 2)
    elif corridor is not None:
        lower = max(mates, key=lambda mate: pedigree.generation[mate])
        edge = x[lower] + math.copysign(_SYMBOL / 2, corridor - x[lower])
        origin = ((corridor + edge) / 2, levels[pedigree.generation[lower]])
    else:
        origin = (plan['centre'], levels[row] if line is None else line)
    father = mother = 0
    for parent in mates:
        if pedigree.sex[parent] == 2:
            mother = int(pedigree.ids[parent])
        else:
            father = int(pedigree.ids[parent])
    born = set(plan['children'])
    return Mating(father=father, mother=mother,
                  consanguineous=pedigree.related(mates),
                  mates=tuple(int(pedigree.ids[mate]) for mate in mates),
                  children=[int(pedigree.ids[child])
                            for child in plan['children']],
                  twins=[(tuple(int(pedigree.ids[twin]) for twin in group),
                          zygosity) for group, zygosity in pedigree.twins
                         if set(group) <= born],
                  childless=pedigree.childless.get(plan['index']),
                  line=line, corridor=corridor, origin=origin,
                  bar=None if plan['above'] is None else
                  levels[plan['above'] + 1] + _SYMBOL / 2
                  + (plan['bar'] + 0.6) * _CHANNEL)


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------


def _draw(layout, labels=None, col=None, fill=None, affected=None,
          deceased=None, carrier=None, proband=None, pregnancy=None,
          count=None, consanguinity=True, cmap='viridis', norm=None,
          colorbar=True, cex=1.0, symbolsize=1.0, generations=False, ax=None,
          figsize=None, title=None):
    """Draw a computed layout; see gen.graph() for the arguments"""
    layout.boxw = layout.boxh = _SYMBOL * symbolsize
    subject = [layout.repeats.get(one, one) for one in layout.ind]
    labels = _labels(subject, labels)
    tags = {one: chr(ord('a') + order % 26) for order, one
            in enumerate(sorted(set(layout.repeats.values())))}
    edges = _mapping(subject, col, 'black')
    marks = {name: _flags(subject, value) for name, value in
             (('deceased', deceased), ('carrier', carrier),
              ('proband', proband), ('pregnancy', pregnancy))}
    marks['count'] = _mapping(subject, count, None)
    faces, colours = _faces(subject, fill, cmap, norm,
                            _flags(subject, affected), edges)
    ax = _axes(layout, figsize) if ax is None else ax
    ax.set_aspect('equal')
    ax.set_axis_off()
    _frame(ax, layout, generations)
    size, page = _typesize(ax, layout) * cex, _furniture(ax) * cex

    where = {identifier: index for index, identifier in enumerate(layout.ind)}
    crossed = _horizontals(layout, where)
    for mating in layout.matings:
        _draw_mating(ax, layout, where, mating, consanguinity, crossed)
    for index, one in enumerate(layout.ind):
        _draw_symbol(ax, layout, index, faces[subject[index]],
                     edges[subject[index]], one in layout.repeats)
        _draw_marks(ax, layout, index, edges[subject[index]], size,
                    {name: flags[subject[index]]
                     for name, flags in marks.items()})
        if labels is not None:
            _draw_label(ax, layout, index,
                        labels[subject[index]] + tags.get(subject[index], ''),
                        edges[subject[index]], size)
    if generations:
        _draw_generations(ax, layout, page * 0.72)
    if title:
        ax.set_title(title, fontsize=page, color='0.1', pad=9)
    if colours is not None and colorbar:
        bar = ax.figure.colorbar(colours, ax=ax, shrink=0.5, fraction=0.025,
                                 pad=0.01)
        bar.outline.set_visible(False)
        bar.ax.tick_params(length=2, labelsize=page * 0.6, colors='0.3')
    return ax


def _axes(layout, figsize):
    """Open a figure whose shape matches the pedigree

    The figure grows with the pedigree rather than being squeezed into a fixed
    page, since a pedigree squeezed far enough becomes a grey smear whatever
    is done to the lettering. Past the point where the drawing would no longer
    fit on any real sheet, that is said plainly instead.
    """
    import matplotlib.pyplot as plt
    width, height = _extent(layout, True)
    unit = min(_INCH, _PAPER / max(width, height, 1.0))
    if figsize is None and max(width, height) * unit > _WIDE:
        warnings.warn(
            f'this pedigree needs {width * unit:.0f} by {height * unit:.0f} '
            'inches to stay legible; consider drawing part of it with '
            'gen.branching() or gen.lineages()', stacklevel=4)
    figure = plt.figure(figsize=figsize or (width * unit, height * unit))
    figure.subplots_adjust(0.005, 0.01, 0.995, 0.99)
    return figure.add_subplot()


def _extent(layout, generations):
    """Width and height the drawing needs, in layout units"""
    margin = _SYMBOL + (1.6 if generations else 0.6)
    width = (float(np.ptp(layout.x)) if layout.x.size else 0.0) + margin + 1.0
    height = (float(np.ptp(layout.levels)) if layout.levels.size else 0.0) \
        + _SYMBOL + _LABEL + 1.4
    return width, height


def _frame(ax, layout, generations):
    """Fit the axes to the drawing, leaving room for the labels and numerals"""
    if not layout.x.size:
        return
    left = 1.6 if generations else 0.7
    ax.set_xlim(float(layout.x.min()) - left - _SYMBOL,
                float(layout.x.max()) + 0.7 + _SYMBOL)
    ax.set_ylim(float(layout.levels.min()) - _SYMBOL - _LABEL - 0.5,
                float(layout.levels.max()) + _SYMBOL + 0.6)


def _typesize(ax, layout):
    """Type size, in points, that keeps the lettering in step with the symbols

    A pedigree of thirty people and one of five hundred are drawn on figures of
    very different sizes, so type fixed in points is either overbearing on the
    one or unreadable on the other. What a label and a mark belong to is the
    symbol they sit on, so they are measured from it, in strict proportion and
    with no floor: lettering too small to read means symbols too small to read,
    and shrinking one without the other only hides that.
    """
    return 0.39 * layout.boxw * _pointsper(ax)


def _pointsper(ax):
    """Points to a layout unit, from the room the axes were given"""
    inches = ax.figure.get_size_inches() * (ax.get_position().width,
                                            ax.get_position().height)
    span = (max(np.ptp(ax.get_xlim()), 1e-9), max(np.ptp(ax.get_ylim()), 1e-9))
    return 72 * min(inches[0] / span[0], inches[1] / span[1])


def _furniture(ax):
    """Type size for the matter that belongs to the page, not to a symbol

    A title and the generation numerals are read from the whole figure at
    arm's length rather than from any one symbol, so they take their size from
    the page. That is what keeps them legible across a chart wide enough that
    the symbols themselves have become small.
    """
    return float(np.clip(2.05 * np.sqrt(np.prod(ax.figure.get_size_inches())),
                         8.0, 30.0))


def _draw_mating(ax, layout, where, mating, consanguinity=True, crossed=()):
    """Draw a mating line and the sibship hanging from it

    Every line here is straight: the lanes reserved during the layout mean a
    descent never has to find its way around anything.
    """
    x, y = layout.x, layout.y
    boxw, boxh = layout.boxw, layout.boxh
    mates = [where[identifier] for identifier in mating.mates]
    if len(mates) == 2:
        left, right = mates
        doubled = mating.consanguineous and consanguinity
        if mating.corridor is not None:
            _mating_line(ax, *_astride(layout, left, right, mating.corridor),
                         doubled, crossed)
        elif mating.line is None:
            _mating_line(ax, [x[left] + boxw / 2, x[right] - boxw / 2],
                         [y[left], y[right]], doubled, crossed)
        else:
            _mating_line(ax, [x[left], x[left], x[right], x[right]],
                         [y[left] - boxh / 2, mating.line, mating.line,
                          y[right] - boxh / 2], doubled, crossed)
    if mating.bar is None:
        _draw_childless(ax, mating)
        return
    children = [where[identifier] for identifier in mating.children]
    span = [x[child] for child in children] + [mating.origin[0]]
    if _lone(layout, where, mating):
        _thread(ax, [(x[children[0]], mating.origin[1]),
                     (x[children[0]], y[children[0]] + boxh / 2)], crossed)
        return
    _line(ax, [min(span), max(span)], [mating.bar] * 2)
    _thread(ax, [mating.origin, (mating.origin[0], mating.bar)], crossed)
    forked = set()
    for group, zygosity in mating.twins:
        members = [where[identifier] for identifier in group]
        if len({int(layout.row[member]) for member in members}) == 1 and \
                mating.bar - y[members[0]] < _ROW_GAP:
            _draw_twins(ax, layout, members, mating.bar, zygosity, crossed)
            forked.update(members)
    for child in children:
        if child not in forked:
            _thread(ax, [(x[child], mating.bar),
                         (x[child], y[child] + boxh / 2)], crossed)


def _horizontals(layout, where):
    """Every horizontal line of the drawing, as (ordinate, left, right)

    Collected before anything is stroked, so that a line of descent knows
    which lines it is going to meet on its way down.
    """
    lines = []
    for mating in layout.matings:
        mates = [where[identifier] for identifier in mating.mates]
        if len(mates) == 2:
            left, right = mates
            if mating.corridor is not None:
                lines += [(layout.y[side], layout.x[side], mating.corridor)
                          for side in (left, right)]
            else:
                lines.append((layout.y[left] if mating.line is None else
                              mating.line, layout.x[left], layout.x[right]))
        if mating.bar is None:
            foot = mating.origin[1] - _ROW_GAP * 0.34
            lines += [(level, mating.origin[0] - _SYMBOL * 0.38,
                       mating.origin[0] + _SYMBOL * 0.38)
                      for level in (foot, foot - _DOUBLE * 1.8)]
            continue
        span = [layout.x[where[child]] for child in mating.children]
        span.append(mating.origin[0])
        if not _lone(layout, where, mating):
            lines.append((mating.bar, min(span), max(span)))
    return lines


def _lone(layout, where, mating):
    """Whether an only child may be reached by a single straight line

    An only child is placed under the midpoint of its parents unless the room
    on its row is contested, in which case it settles a little to one side.
    Drawing that concession as a sibship bar turns a line of descent into a
    staircase, for a displacement no reader could measure, so the line is drawn
    straight instead -- but only where its head would still land on the line
    joining the parents. Where that line is a short arm, as it is for a union
    joined across two rows, a straight drop would start beside it rather than
    on it, and the child is given a proper sibship bar.
    """
    if len(mating.children) != 1:
        return False
    seat = layout.x[where[mating.children[0]]]
    lower, upper = _reach(layout, where, mating)
    return abs(seat - mating.origin[0]) < _SNAP and lower < seat < upper


def _reach(layout, where, mating):
    """The stretch of the line joining the parents a child may hang from"""
    mates = [where[mate] for mate in mating.mates]
    half = layout.boxw / 2
    if len(mates) == 1:
        return layout.x[mates[0]] - half, layout.x[mates[0]] + half
    if mating.corridor is not None:
        lower = min(mates, key=lambda mate: -layout.row[mate])
        edge = layout.x[lower] + math.copysign(
            half, mating.corridor - layout.x[lower])
        return min(edge, mating.corridor), max(edge, mating.corridor)
    seats = sorted(layout.x[mate] for mate in mates)
    if mating.line is not None:
        return seats[0], seats[-1]
    return seats[0] + half, seats[-1] - half


_TURN = np.linspace(0.0, np.pi, 17)


def _thread(ax, points, crossed):
    """Draw a polyline, bridging the horizontals it merely crosses

    A line that simply passes a sibship bar looks exactly like one that joins
    it, which is how a reader comes to believe in parents who are not there.
    Drawing a small bridge at such a crossing is the old convention of the
    draughtsman, and it leaves only the lines that truly meet touching. The
    bridge is raised at right angles to whatever line carries it, so lines of
    descent and the oblique forks of a pair of twins are treated alike.
    """
    path = [np.asarray(points[0], dtype=float)]
    for start, end in zip(points, points[1:]):
        start, end = (np.asarray(point, dtype=float) for point in (start, end))
        reach = np.hypot(*(end - start))
        if not reach:
            continue
        along = (end - start) / reach
        across = np.array([-along[1], along[0]])
        for distance in _meetings(start, end, reach, crossed):
            middle = start + distance * along
            path.extend(middle + _HOP * (np.outer(np.sin(_TURN), across)
                                         - np.outer(np.cos(_TURN), along)))
        path.append(end)
    _line(ax, *zip(*path))


def _meetings(start, end, reach, crossed):
    """How far along a segment it crosses a horizontal line, in order"""
    rise = end[1] - start[1]
    if not rise:
        return []
    distances = []
    for level, left, right in crossed:
        share = (level - start[1]) / rise
        if not 0.0 < share < 1.0:
            continue
        meeting = start[0] + share * (end[0] - start[0])
        if min(left, right) + _HOP < meeting < max(left, right) - _HOP \
                and _HOP < share * reach < reach - _HOP:
            distances.append(share * reach)
    return sorted(distances)


def _draw_twins(ax, layout, members, bar, zygosity, crossed=()):
    """Draw twins as lines of descent meeting at one point of the sibship bar

    Monozygosity is shown by a bar laid between those lines rather than
    between the symbols themselves, and an unknown zygosity by a question
    mark in the same place.
    """
    apex = sum(layout.x[member] for member in members) / len(members)
    feet = [(layout.x[member], layout.y[member] + layout.boxh / 2)
            for member in members]
    for foot, level in feet:
        _thread(ax, [(apex, bar), (foot, level)], crossed)
    across = 0.55
    height = bar + across * (feet[0][1] - bar)
    reach = [apex + across * (foot - apex) for foot, _ in feet]
    if zygosity in ('mz', 'MZ', 'monozygotic', True):
        _line(ax, [min(reach), max(reach)], [height] * 2)
    elif zygosity not in ('dz', 'DZ', 'dizygotic'):
        ax.annotate('?', (apex, height), ha='center', va='center',
                    fontsize=7, color='#3b3b3b', zorder=4,
                    bbox=dict(boxstyle='square,pad=0.05', facecolor='white',
                              edgecolor='none'))


def _draw_childless(ax, mating):
    """Close a union that has no children with the two bars of the convention

    The same mark serves for a couple without children and for an infertile
    one; which it is, if it is known at all, is written beside it.
    """
    x, y = mating.origin
    foot = y - _ROW_GAP * 0.34
    _line(ax, [x, x], [y, foot])
    for level in (foot, foot - _DOUBLE * 1.8):
        _line(ax, [x - _SYMBOL * 0.38, x + _SYMBOL * 0.38], [level] * 2)
    if isinstance(mating.childless, str):
        ax.annotate(mating.childless, (x + _SYMBOL * 0.55, foot), ha='left',
                    va='center', fontsize=7, color='#3b3b3b', zorder=4)


def _astride(layout, left, right, corridor):
    """The path joining two mates left on different rows

    Such a union is drawn as a step: out of the side of the upper mate, down
    the corridor, and into the side of the lower one. Each symbol is left by
    whichever side faces the corridor, and leaving and entering sideways is
    what tells the union apart from a line of descent, which always leaves the
    foot of one symbol and enters the head of another.
    """
    x, y, half = layout.x, layout.y, layout.boxw / 2
    return ([x[left] + np.copysign(half, corridor - x[left]), corridor,
             corridor, x[right] + np.copysign(half, corridor - x[right])],
            [y[left], y[left], y[right], y[right]])


def _apart(mates, x):
    """How much room two mates leave between them, most constrained first"""
    return -abs(x[mates[0]] - x[mates[-1]])


def _corridor(pedigree, mates, x, taken):
    """Where a union spanning two rows steps down from one row to the other

    Between the two mates when that lane is free, which is where a reader looks
    for it; otherwise in the nearest lane that is clear on either side, since a
    corridor squeezed against a symbol reads as a line attached to it.
    Everything the corridor would pass counts against it -- the symbols of
    every row it crosses, the two mates, and the corridors already spoken for
    -- and each is given room on either side. Mates that nearly overlap
    therefore block the lane between them outright, and the corridor goes
    round, which is the only way such a union can be told from a descent.
    """
    left, right = sorted(x[mate] for mate in mates)
    crossed = range(min(pedigree.generation[mate] for mate in mates),
                    max(pedigree.generation[mate] for mate in mates) + 1)
    room = _MATE_GAP * 0.55
    blocked = [(seat - room, seat + room) for seat in
               [x[other] for other in range(pedigree.size)
                if pedigree.generation[other] in crossed and other not in mates]
               + list(taken)]
    blocked += [(x[mate] - _SYMBOL / 2 - _STANDOFF,
                 x[mate] + _SYMBOL / 2 + _STANDOFF) for mate in mates]
    lanes = []
    for start, stop in sorted(blocked):
        if lanes and start <= lanes[-1][1]:
            lanes[-1][1] = max(lanes[-1][1], stop)
        else:
            lanes.append([start, stop])
    middle = (left + right) / 2
    caught = [lane for lane in lanes if lane[0] < middle < lane[1]]
    return middle if not caught else min(caught[0], key=lambda edge:
                                         abs(edge - middle))


def _mating_line(ax, x, y, consanguineous, crossed=()):
    """Draw the line between two mates, doubled if they are related

    Drawing a consanguineous union as a double line is the one pedigree
    convention that carries information the symbols cannot: it says that the
    two sides of the drawing meet again further up. Mates seated side by side
    are joined by the familiar pair of parallel strokes, as the standard
    nomenclature prescribes. A union that had to be joined the long way round
    is doubled along its longest straight run alone -- the corridor of a
    stepped union, the routed line of a couple set too far apart -- since two
    strokes carried round a corner read as two lines rather than as one
    relationship. It is the same mark, laid where the line runs straight.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not consanguineous:
        _thread(ax, list(zip(x, y)), crossed)
        return
    if len(x) == 2:
        _line(ax, x, y + _DOUBLE / 2)
        _line(ax, x, y - _DOUBLE / 2)
        return
    reach = np.hypot(np.diff(x), np.diff(y))
    longest = int(np.argmax(reach))
    for piece, (start, end) in enumerate(zip(zip(x, y), zip(x[1:], y[1:]))):
        if piece != longest:
            _thread(ax, [start, end], crossed)
            continue
        along = (np.array(end) - np.array(start)) / reach[longest]
        across = np.array([-along[1], along[0]]) * (_DOUBLE / 2)
        for side in (-1, 1):
            _thread(ax, [np.array(start) + side * across,
                         np.array(end) + side * across], crossed)


def _draw_symbol(ax, layout, index, face, edge, repeat=False):
    """Draw one individual: square if male, circle if female, else diamond

    A repeat -- the second showing of somebody already drawn elsewhere -- is
    outlined in broken strokes, which is the usual way of saying that the
    symbol stands for a person rather than being one.
    """
    from matplotlib.patches import Circle, Polygon, Rectangle
    x, y, half = layout.x[index], layout.y[index], layout.boxw / 2
    sex = layout.sex[index]
    if sex == 1:
        symbol = Rectangle((x - half, y - half), 2 * half, 2 * half)
    elif sex == 2:
        symbol = Circle((x, y), half)
    else:
        symbol = Polygon([(x, y + half * 1.25), (x + half * 1.25, y),
                          (x, y - half * 1.25), (x - half * 1.25, y)])
    symbol.set(facecolor=face, edgecolor=edge, zorder=3, joinstyle='miter',
               linewidth=0.9 if repeat else 1.0,
               linestyle=(0, (2.2, 1.6)) if repeat else 'solid')
    ax.add_patch(symbol)


def _draw_marks(ax, layout, index, colour, size, marks):
    """Add the marks a pedigree puts on a symbol

    A stroke through the symbol means the individual has died, an arrow from
    the lower left marks a proband, and the middle of the symbol carries, in
    order of precedence, the number of individuals it stands for, a P for a
    pregnancy, or the dot of an obligate carrier.
    """
    from matplotlib.patches import Circle
    x, y, half = layout.x[index], layout.y[index], layout.boxw / 2
    inside = marks['count'] if marks['count'] is not None else \
        'P' if marks['pregnancy'] else None
    if inside is not None:
        ax.annotate(f'{inside:g}' if isinstance(inside, (int, float)) else
                    str(inside), (x, y), ha='center', va='center', zorder=5,
                    fontsize=size * 1.07, color=colour)
    if marks['deceased']:
        ax.plot([x - half * 1.05, x + half * 1.05],
                [y - half * 1.05, y + half * 1.05], color=colour,
                linewidth=1.0, zorder=5, solid_capstyle='round')
    if marks['carrier'] and inside is None:
        ax.add_patch(Circle((x, y), half * 0.24, facecolor=colour,
                            edgecolor='none', zorder=5))
    if marks['proband']:
        ax.annotate('', (x - half * 0.95, y - half * 0.95),
                    (x - half * 2.4, y - half * 2.4), annotation_clip=False,
                    arrowprops=dict(arrowstyle='-|>', color=colour,
                                    linewidth=0.9, shrinkA=0, shrinkB=0,
                                    mutation_scale=7))


def _draw_label(ax, layout, index, text, colour, size):
    """Write the label of one individual, masking the lines behind it

    The patch of white that keeps the label off the lines is kept tight and
    hung clear of the symbol, since a mask wide enough to bite into an outline
    is worse than the lines it was meant to hide.
    """
    if not text:
        return
    below = max(layout.boxh, _SYMBOL) * 0.5 + _LABEL * 0.22
    ax.annotate(text, (layout.x[index], layout.y[index] - below),
                ha='center', va='top', fontsize=size, color=colour,
                zorder=4, annotation_clip=False,
                bbox=dict(boxstyle='square,pad=0.05', facecolor='white',
                          edgecolor='none'))


def _draw_generations(ax, layout, size):
    """Number the generations down the left margin, as pedigree figures do"""
    left = float(layout.x.min()) - 1.5 if layout.x.size else 0.0
    for row, level in enumerate(layout.levels):
        ax.annotate(_roman(row + 1), (left, level), ha='right', va='center',
                    fontsize=size, color='0.45',
                    annotation_clip=False)


def _roman(number):
    """Roman numeral of a generation"""
    text = ''
    for numeral, value in (('L', 50), ('XL', 40), ('X', 10), ('IX', 9),
                           ('V', 5), ('IV', 4), ('I', 1)):
        while number >= value:
            text, number = text + numeral, number - value
    return text


def _line(ax, x, y):
    """Draw one polyline of the pedigree skeleton"""
    ax.plot(x, y, color='#3b3b3b', linewidth=0.85, solid_capstyle='round',
            solid_joinstyle='round', zorder=2)


# -----------------------------------------------------------------------------
# Per-individual options
# -----------------------------------------------------------------------------


def _mapping(ids, value, default):
    """Resolve an option given as a mapping, a sequence, a callable or a scalar

    Returns:
        dict: One value per individual ID.
    """
    if value is None:
        return dict.fromkeys(ids, default)
    if callable(value):
        return {identifier: value(identifier) for identifier in ids}
    if isinstance(value, pd.Series):
        value = value.to_dict()
    if isinstance(value, dict):
        return {identifier: value.get(identifier, default) for identifier in ids}
    if isinstance(value, str) or np.isscalar(value):
        return dict.fromkeys(ids, value)
    if len(value) != len(ids):
        raise ValueError(f'Expected {len(ids)} values, got {len(value)}')
    return dict(zip(ids, value))


def _groups(value):
    """Resolve an option about a tie between individuals

    Ties are given either as the groups themselves or as a mapping from each
    group to what is known about it, which is how zygosity and the reason for
    a union having no children are told apart from the tie itself.

    Returns:
        list: (group, note) pairs, each group a tuple of individual IDs.
    """
    if value is None:
        return []
    if isinstance(value, dict):
        return [(tuple(np.atleast_1d(group)), note)
                for group, note in value.items()]
    return [(tuple(np.atleast_1d(group)), None) for group in value]


def _flags(ids, value):
    """Resolve a yes-or-no option, given as a mapping, a rule or a list of IDs

    Returns:
        dict: True or False for every individual ID.
    """
    if value is None:
        return dict.fromkeys(ids, False)
    if callable(value) or isinstance(value, (dict, pd.Series)):
        return {identifier: bool(marked) for identifier, marked
                in _mapping(ids, value, False).items()}
    marked = list(value)
    if len(marked) == len(ids) and all(isinstance(flag, (bool, np.bool_))
                                       for flag in marked):
        return dict(zip(ids, map(bool, marked)))
    marked = set(marked)
    return {identifier: identifier in marked for identifier in ids}


def _labels(ids, labels):
    """Text drawn under each symbol, or None when labels are turned off"""
    if labels is False:
        return None
    if labels is None or labels is True:
        return {identifier: str(identifier) for identifier in ids}
    return {identifier: '' if text is None else str(text)
            for identifier, text in _mapping(ids, labels, '').items()}


def _faces(ids, fill, cmap, norm, affected, edges):
    """Interior color of each symbol, and the color scale it may come from"""
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    if fill is None:
        return {identifier: edges[identifier] if affected[identifier]
                else 'white' for identifier in ids}, None
    values = _mapping(ids, fill, np.nan)
    if not all(isinstance(value, (int, float, np.number))
               for value in values.values()):
        return {identifier: 'white' if _missing(value) else value
                for identifier, value in values.items()}, None
    finite = [value for value in values.values() if np.isfinite(value)]
    scale = ScalarMappable(Normalize(*(norm or (min(finite, default=0.0),
                                                max(finite, default=1.0)))),
                           cmap)
    scale.set_array([])
    return {identifier: scale.to_rgba(value) if np.isfinite(value) else 'white'
            for identifier, value in values.items()}, scale


def _missing(value):
    """Whether an option was left unset for an individual"""
    return value is None or (isinstance(value, float) and np.isnan(value))
