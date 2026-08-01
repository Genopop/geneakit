"""Tests for the pedigree drawing of geneakit.graph.

The backend has to be chosen before pyplot is imported, which is why the
imports below are not all at the top of the file.
"""

import matplotlib

matplotlib.use('Agg')

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import geneakit as gen  # noqa: E402
from geneakit import genea140, geneaJi  # noqa: E402


def pedigree(ind, father, mother, sex):
    return pd.DataFrame({'ind': ind, 'father': father,
                         'mother': mother, 'sex': sex})


def test_generations_place_children_below_their_parents():
    ped = gen.genealogy(geneaJi)
    layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
    row = dict(zip(layout.ind, layout.row))
    for father, mother, children in layout.families:
        for parent in (father, mother):
            if parent:
                assert all(row[child] > row[parent] for child in children)


def test_mates_share_a_generation_when_the_pedigree_allows_it():
    ped = gen.genealogy(geneaJi)
    layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
    row = dict(zip(layout.ind, layout.row))
    for father, mother, _ in layout.families:
        if father and mother:
            assert row[father] == row[mother]


def test_uncle_and_niece_are_drawn_on_the_same_row():
    frame = pedigree([1, 2, 3, 4, 5, 6, 7],
                     [0, 0, 1, 1, 0, 5, 3],
                     [0, 0, 2, 2, 0, 4, 6],
                     [1, 2, 1, 2, 1, 2, 1])
    layout = gen.PedigreeLayout.from_frame(frame)
    row = dict(zip(layout.ind, layout.row))
    assert row[3] == row[6] == 2


def test_a_cycle_of_descent_and_mating_only_warns():
    frame = pedigree([1, 2, 3, 4, 5], [4, 0, 1, 0, 4],
                     [0, 0, 2, 0, 3], [1, 2, 2, 1, 2])
    with pytest.warns(UserWarning):
        layout = gen.PedigreeLayout.from_frame(frame)
    row = dict(zip(layout.ind, layout.row))
    assert row[1] > row[4] and row[3] > row[1]


def test_an_ancestral_cycle_is_an_error():
    frame = pedigree([1, 2], [2, 1], [0, 0], [1, 1])
    with pytest.raises(ValueError):
        gen.PedigreeLayout.from_frame(frame)


def test_symbols_never_overlap():
    ped = gen.branching(gen.genealogy(genea140), pro=[409033])
    layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
    for row in range(layout.row.max() + 1):
        abscissas = np.sort(layout.x[layout.row == row])
        assert np.all(np.diff(abscissas) >= layout.boxw)


def test_the_layout_is_deterministic():
    frame = gen.genout(gen.genealogy(geneaJi))
    first, second = (gen.PedigreeLayout.from_frame(frame) for _ in range(2))
    assert np.array_equal(first.x, second.x)
    assert np.array_equal(first.nid, second.nid)


def test_the_kinship2_fields_agree_with_the_individual_ones():
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)))
    assert layout.n.sum() == len(layout.ind)
    for individual, row, slot, x in zip(layout.ind, layout.row,
                                        layout.slot, layout.x):
        assert layout.nid[row, slot] == individual
        assert layout.pos[row, slot] == x
    for row, spouses in enumerate(layout.spouse):
        for slot in np.flatnonzero(spouses):
            assert layout.nid[row, slot + 1] != 0


def test_a_pedigree_of_one_and_a_pedigree_of_none():
    layout = gen.PedigreeLayout.from_frame(pedigree([1], [0], [0], [1]))
    assert layout.n.tolist() == [1]
    empty = gen.PedigreeLayout.from_frame(pedigree([], [], [], []))
    assert empty.n.size == 0


def test_graph_draws_every_individual():
    ped = gen.genealogy(geneaJi)
    layout = gen.graph(ped, title='geneaJi', labels=False)
    patches = [patch for patch in layout.ax.patches]
    assert len(patches) == gen.noind(ped)
    plt.close('all')


def test_graph_accepts_variables_keyed_by_individual():
    ped = gen.genealogy(geneaJi)
    inbreeding = gen.f(ped).iloc[:, 0]
    layout = gen.graph(ped, fill=inbreeding, col='0.4',
                       labels={17: 'founder'}, cex=0.8, symbolsize=1.2)
    assert layout.ax is not None
    plt.close('all')


def test_a_lane_is_reserved_for_every_crossing_descent():
    ped = gen.genealogy(geneaJi)
    layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
    where = {identifier: index for index, identifier in enumerate(layout.ind)}
    for mating in layout.matings:
        for child in mating.children:
            index = where[child]
            crossed = [row for row in range(layout.row.max() + 1)
                       if layout.row[index] > row > layout.row[index]
                       - (layout.row[index] - min(layout.row[where[other]]
                                                  for other in mating.children))]
            for row in crossed:
                nearby = layout.x[layout.row == row] - layout.x[index]
                assert np.all(np.abs(nearby) >= layout.boxw)


def test_consanguineous_unions_are_recognized():
    jicaque = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)))
    assert any(mating.consanguineous for mating in jicaque.matings)
    unrelated = gen.PedigreeLayout.from_frame(pedigree(
        [1, 2, 3, 4, 5], [0, 0, 1, 0, 3], [0, 0, 2, 0, 4], [1, 2, 1, 2, 1]))
    assert not any(mating.consanguineous for mating in unrelated.matings)


def test_the_marks_of_the_convention_are_drawn():
    ped = gen.genealogy(geneaJi)
    layout = gen.graph(ped, deceased=[25, 26], carrier=[11], proband=[1],
                       labels=False)
    assert len(layout.ax.patches) == gen.noind(ped) + 1  # one carrier dot
    plt.close('all')


def test_twins_stand_side_by_side_and_share_a_point_of_descent():
    frame = pedigree([1, 2, 3, 4, 5], [0, 0, 1, 1, 1], [0, 0, 2, 2, 2],
                     [1, 2, 1, 2, 1])
    layout = gen.PedigreeLayout.from_frame(frame, twins={(3, 4): 'mz'})
    assert abs(layout.x[2] - layout.x[3]) <= layout.boxw * 2
    mating, = [m for m in layout.matings if m.children]
    assert mating.twins == [((3, 4), 'mz')]


def test_a_union_without_children_is_still_drawn():
    frame = pedigree([1, 2, 3, 4], [0, 0, 1, 0], [0, 0, 2, 0], [1, 2, 1, 2])
    layout = gen.PedigreeLayout.from_frame(frame, childless={(3, 4): 'i'})
    barren = [m for m in layout.matings if m.bar is None]
    assert len(barren) == 1 and barren[0].childless == 'i'
    assert layout.row[layout.ind.tolist().index(3)] == \
        layout.row[layout.ind.tolist().index(4)]


def test_the_middle_of_a_symbol_carries_one_mark_at_a_time():
    ped = gen.genealogy(geneaJi)
    layout = gen.graph(ped, count={17: 4}, pregnancy=[19], carrier=[17],
                       labels=False)
    assert len(layout.ax.patches) == gen.noind(ped)  # no carrier dot under a count
    plt.close('all')


def test_a_descent_bridges_the_bars_it_only_crosses():
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)))
    where = {identifier: index for index, identifier in enumerate(layout.ind)}
    crossed = gen.graph.__globals__['_horizontals'](layout, where)
    bridged = 0
    for mating in layout.matings:
        descents = [(mating.origin[0], mating.origin[1], mating.bar)]
        descents += [(layout.x[where[child]], mating.bar,
                      layout.y[where[child]] + layout.boxh / 2)
                     for child in mating.children]
        for x, top, bottom in descents:
            bridged += sum(1 for level, left, right in crossed
                           if bottom < level < top
                           and min(left, right) < x < max(left, right))
    assert bridged > 0  # the Jicaque pedigree has crossings to bridge


def test_an_only_child_hangs_from_a_single_straight_line():
    frame = pedigree([1, 2, 3, 4, 5], [0, 0, 1, 0, 3], [0, 0, 2, 0, 4],
                     [1, 2, 1, 2, 1])
    layout = gen.PedigreeLayout.from_frame(frame)
    where = {identifier: index for index, identifier in enumerate(layout.ind)}
    for mating in layout.matings:
        if len(mating.children) == 1:
            child = where[mating.children[0]]
            assert abs(layout.x[child] - mating.origin[0]) < layout.boxw


def test_the_convention_never_makes_the_drawing_worse():
    ped = gen.genealogy(geneaJi)
    layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
    for row in range(layout.row.max() + 1):
        abscissas = np.sort(layout.x[layout.row == row])
        assert np.all(np.diff(abscissas) >= layout.boxw)


def test_a_repeat_shortens_the_longest_line_of_descent():
    frame = gen.genout(gen.genealogy(geneaJi))

    def longest(layout):
        where = {one: index for index, one in enumerate(layout.ind)}
        return max(int(layout.row[where[child]])
                   - max(int(layout.row[where[mate]]) for mate in mating.mates)
                   for mating in layout.matings for child in mating.children)
    plain = gen.PedigreeLayout.from_frame(frame)
    repeated = gen.PedigreeLayout.from_frame(frame, repeat=3)
    assert repeated.repeats and longest(repeated) < longest(plain)
    assert set(repeated.repeats.values()) <= set(plain.ind.tolist())


def test_descent_alignment_makes_the_rows_true_generations():
    frame = gen.genout(gen.genealogy(geneaJi))
    layout = gen.PedigreeLayout.from_frame(frame, align='descent', repeat=None)
    where = {one: index for index, one in enumerate(layout.ind)}
    for mating in layout.matings:
        for child in mating.children:
            assert layout.row[where[child]] == 1 + max(
                layout.row[where[mate]] for mate in mating.mates)


def test_lettering_keeps_step_with_the_symbols():
    """Labels are a fixed fraction of the symbol, whatever the pedigree's size"""
    inner = gen.graph.__globals__
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        drawings = [gen.graph(gen.genealogy(geneaJi), labels=False),
                    gen.graph(gen.branching(gen.genealogy(genea140),
                                            pro=[409033]), labels=False)]
    ratios = [inner['_typesize'](layout.ax, layout)
              / (layout.boxw * inner['_pointsper'](layout.ax))
              for layout in drawings]
    assert abs(ratios[0] - ratios[1]) < 1e-9
    plt.close('all')


def test_a_pedigree_too_large_to_draw_whole_says_so():
    """The symbols keep their size and the figure grows, until it cannot"""
    ped = gen.branching(gen.genealogy(genea140), pro=[409033])
    with pytest.warns(UserWarning, match='branching'):
        layout = gen.graph(ped, labels=False)
    symbol = layout.boxw * gen.graph.__globals__['_pointsper'](layout.ax)
    assert symbol > 6.0
    plt.close('all')


def test_the_convention_never_separates_a_twice_married_person():
    frame = gen.genout(gen.genealogy(geneaJi))
    layout = gen.PedigreeLayout.from_frame(frame)
    assert not any(mating.line is not None for mating in layout.matings)


def test_a_union_joined_the_long_way_is_doubled_on_its_straightest_run():
    """The standard mark is two strokes where there would be one, not a symbol

    Doubling only the longest straight piece keeps the convention recognisable
    and puts it clear of the corners and of the line the children hang from.
    """
    inner = gen.graph.__globals__
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    where = {one: index for index, one in enumerate(layout.ind)}
    bent = [mating for mating in layout.matings
            if mating.consanguineous and mating.corridor is not None]
    assert bent
    for mating in bent:
        left, right = (where[mate] for mate in mating.mates)
        across, along = inner['_astride'](layout, left, right,
                                          mating.corridor)
        reaches = np.hypot(np.diff(across), np.diff(along))
        longest = int(np.argmax(reaches))
        # the corridor, running between the two rows, is the piece to double
        assert abs(across[longest] - across[longest + 1]) < 1e-9
        assert reaches[longest] > inner['_DOUBLE'] * 3


def test_side_by_side_mates_keep_the_plain_double_line():
    ped = gen.genealogy(geneaJi)
    layout = gen.PedigreeLayout.from_frame(gen.genout(ped))
    doubled = [mating for mating in layout.matings if mating.consanguineous]
    assert doubled and all(mating.line is None for mating in doubled)


def test_mates_of_two_rows_are_joined_side_to_side():
    """Such a union must leave and enter sideways, and never stack its mates"""
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    where = {one: index for index, one in enumerate(layout.ind)}
    stepped = [mating for mating in layout.matings
               if mating.corridor is not None]
    assert stepped
    for mating in stepped:
        seats = [layout.x[where[mate]] for mate in mating.mates]
        assert abs(seats[0] - seats[1]) > layout.boxw
        assert min(abs(mating.corridor - seat) for seat in seats) \
            > layout.boxw / 2


def test_no_two_unions_step_down_the_same_lane():
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    lanes = sorted(mating.corridor for mating in layout.matings
                   if mating.corridor is not None)
    assert all(second - first > layout.boxw
               for first, second in zip(lanes, lanes[1:]))


def test_a_corridor_is_clear_however_the_pedigree_was_read():
    """The same pedigree in a different row order must be laid out as well"""
    frames = [gen.genout(gen.genealogy(geneaJi)),
              pd.read_csv(geneaJi, sep='\t').iloc[::-1].reset_index(drop=True)]
    for frame in frames:
        layout = gen.PedigreeLayout.from_frame(frame, align='descent')
        where = {one: index for index, one in enumerate(layout.ind)}
        for mating in layout.matings:
            if mating.corridor is None:
                continue
            rows = [layout.row[where[mate]] for mate in mating.mates]
            seats = layout.x[np.isin(layout.row,
                                     list(range(min(rows), max(rows) + 1)))]
            assert np.all(np.abs(seats - mating.corridor) > layout.boxw * 0.8)


def test_a_label_never_bites_into_the_outline_of_its_symbol():
    """The white patch behind a label must clear the symbol it belongs to"""
    layout = gen.graph(gen.genealogy(geneaJi), generations=True)
    axes = layout.ax
    for index in range(len(layout.ind)):
        foot = axes.transData.transform(
            (layout.x[index], layout.y[index] - layout.boxh / 2))[1]
        head = axes.transData.transform(
            (layout.x[index],
             layout.y[index] - (layout.boxh * 0.5 + 0.46 * 0.22)))[1]
        assert foot - head > 2.0
    plt.close('all')


def test_children_of_a_stepped_union_hang_from_its_arm():
    """Not from the corner, where the drop would look like the corridor going on"""
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    where = {one: index for index, one in enumerate(layout.ind)}
    stepped = [mating for mating in layout.matings
               if mating.corridor is not None and mating.children]
    assert stepped
    for mating in stepped:
        lower = max((where[mate] for mate in mating.mates),
                    key=lambda index: layout.row[index])
        assert min(abs(mating.origin[0] - mating.corridor),
                   abs(mating.origin[0] - layout.x[lower])) > 0.05
        assert min(mating.corridor, layout.x[lower]) < mating.origin[0] \
            < max(mating.corridor, layout.x[lower])


def test_every_crossing_in_the_drawing_is_bridged():
    """No two lines may simply cut through one another"""
    inner = gen.graph.__globals__
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    where = {one: index for index, one in enumerate(layout.ind)}
    horizontals = inner['_horizontals'](layout, where)
    segments = []
    for mating in layout.matings:
        mates = [where[mate] for mate in mating.mates]
        if len(mates) == 2 and mating.corridor is not None:
            across, along = inner['_astride'](layout, *mates, mating.corridor)
            segments += list(zip(zip(across, along),
                                 zip(across[1:], along[1:])))
        if mating.bar is None:
            continue
        segments.append((tuple(mating.origin),
                         (mating.origin[0], mating.bar)))
        for child in mating.children:
            segments.append(((layout.x[where[child]], mating.bar),
                             (layout.x[where[child]],
                              layout.y[where[child]] + layout.boxh / 2)))
    for first in segments:
        for second in segments:
            meeting = _crossing(first, second)
            if meeting is None:
                continue
            assert any(abs(level - meeting[1]) < 1e-6
                       and min(left, right) < meeting[0] < max(left, right)
                       for level, left, right in horizontals)


def _crossing(first, second):
    """Where two segments cut through one another, if they properly do"""
    (ax, ay), (bx, by) = first
    (cx, cy), (dx, dy) = second
    slant = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx)
    if abs(slant) < 1e-12:
        return None
    along = ((cx - ax) * (dy - cy) - (cy - ay) * (dx - cx)) / slant
    across = ((cx - ax) * (by - ay) - (cy - ay) * (bx - ax)) / slant
    if 0.02 < along < 0.98 and 0.02 < across < 0.98:
        return ax + along * (bx - ax), ay + along * (by - ay)
    return None


def test_a_corridor_keeps_clear_of_every_row_it_crosses():
    """Including of its own mates, so it never looks attached to a symbol"""
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    where = {one: index for index, one in enumerate(layout.ind)}
    stepped = [mating for mating in layout.matings
               if mating.corridor is not None]
    assert stepped
    for mating in stepped:
        rows = [layout.row[where[mate]] for mate in mating.mates]
        crossed = range(min(rows), max(rows) + 1)
        seats = layout.x[np.isin(layout.row, list(crossed))]
        assert np.all(np.abs(seats - mating.corridor) > layout.boxw * 0.8)


def test_every_line_of_descent_starts_on_the_line_joining_the_parents():
    """A descent that begins beside its mating line looks like an orphan"""
    inner = gen.graph.__globals__
    for keywords in ({}, {'align': 'descent'}):
        layout = gen.PedigreeLayout.from_frame(
            gen.genout(gen.genealogy(geneaJi)), **keywords)
        where = {one: index for index, one in enumerate(layout.ind)}
        for mating in layout.matings:
            if mating.bar is None:
                continue
            lower, upper = inner['_reach'](layout, where, mating)
            if inner['_lone'](layout, where, mating):
                head = layout.x[where[mating.children[0]]]
            else:
                head = mating.origin[0]
            assert lower - 1e-9 <= head <= upper + 1e-9


def test_an_only_child_of_a_stepped_union_gets_a_sibship_bar():
    """Its parents' line is a short arm, too short to drop straight from"""
    inner = gen.graph.__globals__
    layout = gen.PedigreeLayout.from_frame(gen.genout(gen.genealogy(geneaJi)),
                                           align='descent')
    where = {one: index for index, one in enumerate(layout.ind)}
    stepped = [mating for mating in layout.matings
               if mating.corridor is not None and len(mating.children) == 1]
    for mating in stepped:
        lower, upper = inner['_reach'](layout, where, mating)
        seat = layout.x[where[mating.children[0]]]
        if not lower < seat < upper:
            assert not inner['_lone'](layout, where, mating)
