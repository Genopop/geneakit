"""Tests for the dictionary-like editing interface of a pedigree.

The two invariants every test below leans on are:

  * `ids` is topologically ordered -- a parent always comes before their
    children;
  * an individual's rank is its position in `ids`, so ranks are always the
    dense range 0..N-1.
"""

import random

import numpy as np
import pandas as pd
import pytest

import geneakit as gen


def pedigree(rows):
    """rows: list of (ind, father, mother, sex)."""
    frame = pd.DataFrame(rows, columns=['ind', 'father', 'mother', 'sex'])
    return gen.genealogy(frame)


def ranks(ped):
    return [ped[identifier].rank for identifier in ped.keys()]


def check_invariants(ped):
    identifiers = list(ped.keys())
    assert ranks(ped) == list(range(len(identifiers))), \
        'ranks must be the dense range 0..N-1, in ids order'
    assert len(set(identifiers)) == len(identifiers)
    position = {identifier: index for index, identifier in
                enumerate(identifiers)}
    for identifier in identifiers:
        individual = ped[identifier]
        for parent in (individual.father, individual.mother):
            if parent.ind:
                assert position[parent.ind] < position[identifier], \
                    f'{parent.ind} must come before their child {identifier}'


# --------------------------------------------------------------------------
# The motivating case: a correction that invalidates the existing order
# --------------------------------------------------------------------------

def test_a_new_parent_added_last_is_moved_before_their_child():
    # 1 is the father of 2; 3 is added afterwards, so it comes last.
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    ped[3] = (0, 0, 1)
    assert list(ped.keys()) == [1, 2, 3]
    # The genealogy is corrected: 3, not 1, is really the father of 2.
    ped[2] = (3, 0, 1)
    assert list(ped.keys()) == [1, 3, 2]
    check_invariants(ped)


def test_a_new_parent_drags_their_own_ancestors_along():
    # 9 -> 3 sits after 2, and 3 becomes the father of 2, so both 9 and 3
    # have to end up before 2.
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    ped[9] = (0, 0, 1)
    ped[3] = (9, 0, 1)
    assert list(ped.keys()) == [1, 2, 9, 3]
    ped[2] = (3, 0, 1)
    assert list(ped.keys()) == [1, 9, 3, 2]
    check_invariants(ped)


def test_a_child_of_the_new_parent_stays_after_them():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 2), (3, 0, 0, 1)])
    ped[4] = (3, 2, 1)          # 4 is a descendant of 2 and 3
    ped[2] = (3, 0, 2)          # and now 3 also becomes the father of 2
    check_invariants(ped)
    assert ped[3].rank < ped[2].rank < ped[4].rank


def test_only_the_affected_individuals_are_moved():
    # 100 unrelated founders, then a correction that concerns two of them.
    ped = pedigree([(index, 0, 0, 1) for index in range(1, 101)])
    before = {identifier: ped[identifier].rank for identifier in ped.keys()}
    ped[10] = (90, 0, 1)        # 90 must move in front of 10
    check_invariants(ped)
    moved = {identifier for identifier in ped.keys()
             if ped[identifier].rank != before[identifier]}
    # 90 has no ancestors and 10 no descendants, so the correction is a
    # straight swap: the 79 individuals sitting between them do not move,
    # and no rank outside the pair is disturbed.
    assert moved == {10, 90}
    assert ped[90].rank < ped[10].rank


def test_the_moved_set_is_the_constrained_set():
    # 200 founders; 150 gets a child (151) and 50 gets a parent (49), so
    # the repair below has to carry a descendant and an ancestor with it.
    ped = pedigree([(index, 0, 0, 1) for index in range(1, 201)])
    ped[201] = (50, 0, 1)       # 201 is a descendant of 50
    ped[150] = (149, 0, 1)      # 149 is an ancestor of 150
    before = {identifier: ped[identifier].rank for identifier in ped.keys()}
    ped[50] = (150, 0, 1)       # 150 must come before 50
    check_invariants(ped)
    moved = {identifier for identifier in ped.keys()
             if ped[identifier].rank != before[identifier]}
    # 150 and its ancestor 149 slide in front of 50, and 50 slides behind
    # them. 201 already sat after all of them, so it does not have to move,
    # and neither does anybody else: 4 of 201 individuals are touched.
    assert moved == {50, 149, 150}
    assert ped[149].rank < ped[150].rank < ped[50].rank < ped[201].rank


# --------------------------------------------------------------------------
# Cycles
# --------------------------------------------------------------------------

def test_an_individual_cannot_become_their_own_ancestor():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1), (3, 2, 0, 1)])
    with pytest.raises(ValueError, match='descendant'):
        ped[1] = (3, 0, 1)      # 3 is a great-grandchild of 1
    check_invariants(ped)


def test_an_individual_cannot_be_their_own_parent():
    ped = pedigree([(1, 0, 0, 1)])
    with pytest.raises(ValueError, match='own father'):
        ped[1] = (1, 0, 1)


def test_the_same_individual_cannot_be_both_parents():
    ped = pedigree([(1, 0, 0, 1), (2, 0, 0, 1)])
    with pytest.raises(ValueError, match='both'):
        ped[2] = (1, 1, 1)


# --------------------------------------------------------------------------
# A rejected edit must change nothing at all
# --------------------------------------------------------------------------

@pytest.mark.parametrize('record, error', [
    ((3, 0, 1), ValueError),        # 3 is a descendant of 1
    ((0, 4, 1), KeyError),          # 4 is not in the pedigree
    ((0, 0, 7), ValueError),        # 7 is not a sex code
    ((1, 1, 1), ValueError),        # both parents at once
])
def test_a_rejected_edit_leaves_the_pedigree_untouched(record, error):
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 2), (3, 2, 0, 1)])
    before = gen.genout(ped)
    order = list(ped.keys())
    with pytest.raises(error):
        ped[1] = record
    pd.testing.assert_frame_equal(gen.genout(ped), before)
    assert list(ped.keys()) == order
    check_invariants(ped)


def test_a_parent_of_the_wrong_sex_is_refused():
    ped = pedigree([(1, 0, 0, 2), (2, 0, 0, 1)])
    with pytest.raises(ValueError, match='female'):
        ped[2] = (1, 0, 1)
    with pytest.raises(ValueError, match='male'):
        ped[1] = (0, 2, 2)


def test_a_parent_cannot_be_given_a_sex_that_denies_their_role():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    with pytest.raises(ValueError, match='cannot be marked female'):
        ped[1] = (0, 0, 2)
    check_invariants(ped)


# --------------------------------------------------------------------------
# Removal
# --------------------------------------------------------------------------

def test_removal_keeps_the_ranks_dense():
    ped = pedigree([(index, 0, 0, 1) for index in range(1, 11)])
    del ped[5]
    del ped[1]
    assert list(ped.keys()) == [2, 3, 4, 6, 7, 8, 9, 10]
    check_invariants(ped)


def test_removal_orphans_the_children_on_that_side_only():
    ped = pedigree([(1, 0, 0, 1), (2, 0, 0, 2), (3, 1, 2, 1)])
    del ped[1]
    assert ped[3].father.ind == 0
    assert ped[3].mother.ind == 2
    check_invariants(ped)


def test_a_removed_individual_is_gone():
    ped = pedigree([(1, 0, 0, 1)])
    del ped[1]
    assert 1 not in ped
    assert len(ped) == 0
    with pytest.raises(KeyError):
        del ped[1]


def test_batch_removal_matches_removing_one_at_a_time():
    rows = [(1, 0, 0, 1), (2, 0, 0, 2), (3, 1, 2, 1), (4, 1, 2, 2),
            (5, 3, 4, 1)]
    one_by_one = pedigree(rows)
    for identifier in (1, 3):
        del one_by_one[identifier]
    at_once = pedigree(rows)
    at_once.remove([3, 1])
    pd.testing.assert_frame_equal(gen.genout(at_once), gen.genout(one_by_one))
    check_invariants(at_once)


def test_a_batch_removal_with_a_missing_id_removes_nothing():
    ped = pedigree([(1, 0, 0, 1), (2, 0, 0, 2)])
    with pytest.raises(KeyError):
        ped.remove([1, 42])
    assert list(ped.keys()) == [1, 2]


# --------------------------------------------------------------------------
# The batch interface
# --------------------------------------------------------------------------

def test_a_batch_may_name_parents_it_introduces_itself():
    ped = pedigree([(1, 0, 0, 1)])
    # 2's parents are defined later in the same batch, and 3 is defined
    # after the child that refers to it.
    ped.update({2: (3, 4, 1), 3: (0, 0, 1), 4: (0, 0, 2)})
    check_invariants(ped)
    assert ped[2].father.ind == 3
    assert ped[2].mother.ind == 4


def test_a_batch_and_a_rebuild_agree():
    ped = pedigree([(1, 0, 0, 1)])
    ped.update({2: (3, 4, 1), 3: (0, 0, 1), 4: (0, 0, 2), 1: (3, 4, 1)})
    rebuilt = pedigree([(3, 0, 0, 1), (4, 0, 0, 2), (1, 3, 4, 1),
                        (2, 3, 4, 1)])
    pd.testing.assert_frame_equal(
        gen.genout(ped).sort_values('ind').reset_index(drop=True),
        gen.genout(rebuilt).sort_values('ind').reset_index(drop=True))


def test_a_batch_accepts_a_list_of_records():
    ped = pedigree([(1, 0, 0, 1)])
    ped.update([(2, 0, 0, 2), (3, 1, 2, 1)])
    check_invariants(ped)
    assert ped[3].father.ind == 1 and ped[3].mother.ind == 2


def test_the_last_record_of_a_batch_wins():
    ped = pedigree([(1, 0, 0, 1), (2, 0, 0, 2)])
    ped.update([(3, 1, 0, 1), (3, 0, 2, 1)])
    assert ped[3].father.ind == 0
    assert ped[3].mother.ind == 2


def test_a_batch_that_would_close_a_cycle_is_refused_whole():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    before = gen.genout(ped)
    with pytest.raises(ValueError, match='own ancestor'):
        ped.update({1: (3, 0, 1), 3: (2, 0, 1), 4: (0, 0, 2)})
    pd.testing.assert_frame_equal(gen.genout(ped), before)
    assert 4 not in ped
    check_invariants(ped)


def test_a_batch_sees_sex_corrections_made_in_the_same_batch():
    ped = pedigree([(1, 0, 0, 2), (2, 0, 0, 1)])
    # 1 is recorded female, so it could not be a father -- unless the same
    # batch also corrects its sex, which it does.
    ped.update({1: (0, 0, 1), 2: (1, 0, 1)})
    assert ped[2].father.ind == 1
    check_invariants(ped)


def test_a_batch_cannot_break_a_child_it_does_not_mention():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    with pytest.raises(ValueError, match='female'):
        ped.update({1: (0, 0, 2)})      # 1 is the father of 2
    check_invariants(ped)


def test_a_batch_keeps_the_previous_order_where_it_can():
    ped = pedigree([(index, 0, 0, 1) for index in range(1, 6)])
    ped.update({6: (0, 0, 1)})
    assert list(ped.keys()) == [1, 2, 3, 4, 5, 6]


def test_an_empty_batch_is_a_no_op():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    ped.update({})
    assert list(ped.keys()) == [1, 2]


# --------------------------------------------------------------------------
# The mapping interface itself
# --------------------------------------------------------------------------

def test_keys_values_and_items_are_in_the_same_order():
    ped = pedigree([(1, 0, 0, 1), (2, 0, 0, 2), (3, 1, 2, 1)])
    ped[4] = (0, 0, 1)
    ped[3] = (4, 2, 1)          # forces a reordering
    keys = list(ped.keys())
    assert [individual.ind for individual in ped.values()] == keys
    assert [identifier for identifier, _ in ped.items()] == keys
    assert [individual.ind for _, individual in ped.items()] == keys
    assert list(ped) == keys


def test_children_are_listed_in_pedigree_order():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1), (3, 1, 0, 1), (4, 1, 0, 1)])
    assert [child.ind for child in ped[1].children] == [2, 3, 4]
    del ped[3]
    assert [child.ind for child in ped[1].children] == [2, 4]


def test_length_and_membership():
    ped = pedigree([(1, 0, 0, 1), (2, 1, 0, 1)])
    assert len(ped) == 2
    assert 1 in ped and 42 not in ped
    ped[3] = (0, 0, 1)
    assert len(ped) == 3


# --------------------------------------------------------------------------
# Property tests: any sequence of edits keeps the pedigree consistent, and
# an edited pedigree computes exactly like the same pedigree rebuilt from
# scratch.
# --------------------------------------------------------------------------

def random_frame(size, seed):
    generator = random.Random(seed)
    rows, males, females = [], [], []
    for identifier in range(1, size + 1):
        sex = generator.choice([1, 2])
        father = (generator.choice(males)
                  if males and generator.random() < 0.75 else 0)
        mother = (generator.choice(females)
                  if females and generator.random() < 0.75 else 0)
        rows.append((identifier, father, mother, sex))
        (males if sex == 1 else females).append(identifier)
    return rows


def build_by_editing(rows, seed):
    """Add everyone as a founder in random order, then fill the parents in.

    This is the natural shape of a data-correction session, and every link
    it sets up runs against the order the pedigree happens to be in.
    """
    generator = random.Random(seed + 1)
    ped = pedigree([(rows[0][0], 0, 0, rows[0][3])])
    shuffled = rows[1:]
    generator.shuffle(shuffled)
    for identifier, _, _, sex in shuffled:
        ped[identifier] = (0, 0, sex)
    for identifier, father, mother, sex in rows:
        ped[identifier] = (father, mother, sex)
    return ped


def analytics(ped):
    probands = sorted(gen.pro(ped))
    founders = sorted(gen.founder(ped))
    return {
        'genout': gen.genout(ped).sort_values('ind').to_numpy().tolist(),
        'probands': probands,
        'founders': founders,
        'depth': gen.depth(ped),
        'f': np.round(np.asarray(gen.f(ped, pro=probands)).ravel(), 12
                      ).tolist(),
        'phi': np.round(np.asarray(gen.phi(ped, pro=probands)), 12).tolist(),
        'gc': np.round(np.asarray(
            gen.gc(ped, pro=probands, ancestors=founders)), 12).tolist(),
    }


@pytest.mark.parametrize('seed', range(12))
def test_editing_reproduces_a_pedigree_built_from_a_frame(seed):
    rows = random_frame(25, seed)
    edited = build_by_editing(rows, seed)
    check_invariants(edited)
    assert analytics(edited) == analytics(pedigree(rows))


@pytest.mark.parametrize('seed', range(12))
def test_a_batch_reproduces_a_pedigree_built_from_a_frame(seed):
    rows = random_frame(25, seed)
    generator = random.Random(seed + 2)
    shuffled = list(rows)
    generator.shuffle(shuffled)
    edited = gen.genealogy(pd.DataFrame(
        [rows[0]], columns=['ind', 'father', 'mother', 'sex']))
    edited.update([record for record in shuffled])
    check_invariants(edited)
    assert analytics(edited) == analytics(pedigree(rows))


@pytest.mark.parametrize('seed', range(12))
def test_random_edits_never_break_the_invariants(seed):
    generator = random.Random(seed + 3)
    rows = random_frame(20, seed)
    ped = pedigree(rows)
    alive = [identifier for identifier, _, _, _ in rows]
    for _ in range(120):
        action = generator.random()
        if action < 0.2 and len(alive) > 2:
            del ped[alive.pop(generator.randrange(len(alive)))]
        elif action < 0.4:
            identifier = max(alive) + 1
            ped[identifier] = (0, 0, generator.choice([1, 2]))
            alive.append(identifier)
        else:
            identifier = generator.choice(alive)
            father = generator.choice(alive + [0])
            mother = generator.choice(alive + [0])
            sex = generator.choice([0, 1, 2])
            try:
                ped[identifier] = (father, mother, sex)
            except (ValueError, KeyError):
                pass            # an inconsistent correction; must be a no-op
        check_invariants(ped)


@pytest.mark.parametrize('seed', range(6))
def test_a_pedigree_survives_a_round_trip_through_editing(seed):
    rows = random_frame(30, seed)
    ped = pedigree(rows)
    # Strip every parent link, then put every one of them back.
    for identifier, _, _, sex in rows:
        ped[identifier] = (0, 0, sex)
    assert gen.noind(ped) == len(rows)
    for identifier, father, mother, sex in rows:
        ped[identifier] = (father, mother, sex)
    check_invariants(ped)
    assert analytics(ped) == analytics(pedigree(rows))
