from __future__ import annotations

import numpy as np
import pytest

from stark.core.block import Block, BlockBasis
from stark.engines.carriers import CarrierNative, CarrierNumpy


def test_block_basis_lifts_translation_bases() -> None:
    basis_a = CarrierNative([0.0, 0.0]).basis
    basis_b = CarrierNative([0.0]).basis
    block_basis = BlockBasis([basis_a, basis_b])
    output = Block([[0.0, 0.0], [0.0]])

    assert block_basis.dimension == 3
    assert block_basis.offsets == (0, 2, 3)
    assert block_basis.local_index(2) == (1, 0)

    vector = block_basis.vector(1, output)

    assert vector is output
    assert vector[0] == [0.0, 1.0]
    assert vector[1] == [0.0]
    assert block_basis.coordinate(1, vector) == pytest.approx(1.0)
    assert block_basis.coordinate(2, vector) == pytest.approx(0.0)


def test_block_basis_analyses_and_synthesizes_blocks() -> None:
    basis_a = CarrierNative([0.0, 0.0]).basis
    basis_b = CarrierNative((0.0, 0.0, 0.0)).basis
    block_basis = BlockBasis([basis_a, basis_b])
    coordinates = [0.0] * block_basis.dimension

    block_basis.coordinates(Block([[1.0, 2.0], (3.0, 4.0, 5.0)]), coordinates)

    assert coordinates == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])

    output = Block([[0.0, 0.0], (0.0, 0.0, 0.0)])
    block = block_basis.synthesize([5.0, 4.0, 3.0, 2.0, 1.0], output)

    assert block is output
    assert block[0] == [5.0, 4.0]
    assert block[1] == (3.0, 2.0, 1.0)


def test_block_basis_preserves_return_style_entries() -> None:
    basis = CarrierNumpy(np.zeros(2)).basis
    block_basis = BlockBasis([basis])
    output = Block([np.zeros(2)])

    vector = block_basis.vector(1, output)

    assert vector is output
    np.testing.assert_allclose(vector[0], [0.0, 1.0])


def test_block_basis_keeps_index_check_but_trusts_prepared_block_size() -> None:
    block_basis = BlockBasis([CarrierNative([0.0]).basis])
    coordinates = [0.0]

    block_basis.coordinates(Block([[2.0], [99.0]]), coordinates)

    assert coordinates == pytest.approx([2.0])

    with pytest.raises(IndexError):
        block_basis.local_index(1)
