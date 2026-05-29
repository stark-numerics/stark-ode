from typing import Protocol, TypeVar

from stark.contracts.translation import Translation, TranslationType

class BlockLike(Protocol[TranslationType]):
    size: int

    def __len__(self) -> int:
        ...

    def __getitem__(self, index: int) -> TranslationType:
        ...

    def __setitem__(self, index: int, value: TranslationType) -> None:
        ...

    def replace(self, other: "BlockLike[TranslationType]") -> None:
        ...

    def norm(self) -> float:
        ...


class BlockOperatorLike(Protocol[TranslationType]):
    size: int

    def reset(self) -> None:
        ...

    def __call__(
        self,
        source: BlockLike[TranslationType],
        target: BlockLike[TranslationType],
    ) -> None:
        ...