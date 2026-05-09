from pathlib import Path


EVAL_HOLE_ID_RANGES = ((1, 17), (54, 71))
EVAL_HOLE_IDS = frozenset(
    hole_id
    for start, end in EVAL_HOLE_ID_RANGES
    for hole_id in range(start, end + 1)
)


def hole_id_from_path(path: Path) -> int:
    """Extract numeric hole id from filenames such as 000001.png."""
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(f"Expected numeric hole filename, got {path.name!r}") from exc


def split_pairs_by_hole_id(
    image_paths: list[Path],
    mask_paths: list[Path],
    eval_hole_ids: frozenset[int] = EVAL_HOLE_IDS,
) -> tuple[tuple[list[Path], list[Path]], tuple[list[Path], list[Path]]]:
    """Split image/mask pairs into train and eval sets using global hole ids."""
    mask_by_name = {path.name: path for path in mask_paths}
    train_pairs = []
    eval_pairs = []

    for image_path in image_paths:
        try:
            pair = (image_path, mask_by_name[image_path.name])
        except KeyError as exc:
            raise FileNotFoundError(f"No mask found for image {image_path.name}") from exc

        if hole_id_from_path(image_path) in eval_hole_ids:
            eval_pairs.append(pair)
        else:
            train_pairs.append(pair)

    if not train_pairs:
        raise ValueError("Global split produced an empty train set.")
    if not eval_pairs:
        raise ValueError("Global split produced an empty eval set.")

    train_images, train_masks = zip(*train_pairs)
    eval_images, eval_masks = zip(*eval_pairs)

    return (list(train_images), list(train_masks)), (list(eval_images), list(eval_masks))
