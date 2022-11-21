import argparse
import pathlib
from dataclasses import dataclass
from typing import Generator, Tuple, Dict, Any, Union

import cv2
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json


@dataclass
class Base:
    def __call__(self, *args, **kwargs) -> Dict[str, Any]:
        return self.__dict__


@dataclass
class Annotation(Base):
    id: int = 1
    name: str = 'car'
    bbox: Tuple[float, float, float, float] = tuple()


@dataclass
class Frame:
    id: int = 1
    file_path: str = ''
    height: int = 640
    width: int = 640


@dataclass
class Video(Base):
    file_path: str = ''
    frames: Union[Dict[int, Frame], None] = None
    annotations: Union[Dict[int, Annotation], None] = None


def get_frames(video_path: pathlib.Path, frame_extension='jpg') -> Union[Dict[int, Frame]]:
    video_capture = cv2.VideoCapture(str(video_path))
    success, image = video_capture.read()
    dest_dir: pathlib.Path = pathlib.Path(video_path.parents[0] / video_path.stem)
    dest_dir.mkdir(parents=True, exist_ok=True)
    frame: int = 0
    frames: Union[Dict[int, Frame]] = {}
    while success:
        frame += 1
        dest_path: str = str(dest_dir / f'{frame}.{frame_extension}')
        cv2.imwrite(dest_path, image)
        height, width, _ = cv2.imread(dest_path).shape
        frames[frame] = Frame(
            id=frame,
            file_path=dest_path,
            height=height,
            width=width,
        )
        success, image = video_capture.read()

    video_capture.release()
    cv2.destroyAllWindows()
    return frames


def get_annotations(annotation_path: pathlib.Path) -> Union[Dict[int, Annotation]]:
    annotations: Union[Dict[int, Annotation]] = {}
    with open(annotation_path, 'r') as annotation:
        lines: list[str] = annotation.readlines()[1::]
    for line in lines:
        if line.strip() == '':
            continue
        processed_line: list[str] = line.split('\t')
        annotations[int(processed_line[0])] = Annotation(
            id=int(processed_line[0]),
            bbox=(
                float(processed_line[2]),
                float(processed_line[3]),
                float(processed_line[4]),
                float(processed_line[5])
            )
        )
    return annotations


def get_video(video_path: pathlib.Path, annotation_path: pathlib.Path) -> Video:
    print('Working on frames..', end=' ')
    frames = get_frames(video_path)
    print(' | Done!')
    print('Working on annotations..', end=' ')
    annotations = get_annotations(annotation_path)
    print(' | Done!')
    video = Video(
        file_path=str(video_path),
        frames=frames,
        annotations=annotations
    )
    print('######')
    return video


def load_dataset(dataset_dir: str) -> Generator[Video, None, None]:
    for annotation, video in zip(
        sorted(pathlib.Path(dataset_dir).glob('*.txt')),
        sorted(pathlib.Path(dataset_dir).glob('*.MOV'))
    ):
        yield get_video(video, annotation)


def main(dataset_dir: str, save_path: str) -> None:
    dataset_generator = load_dataset(dataset_dir)
    coco = Coco()
    coco.add_category(CocoCategory(id=0, name='car'))
    for video in dataset_generator:
        for i, annotation in video.annotations.items():
            if i not in video.frames:
                continue
            frame = video.frames[i]
            coco_image = CocoImage(file_name=frame.file_path, height=frame.height, width=frame.width)
            coco_image.add_annotation(
                CocoAnnotation(
                    bbox=annotation.bbox,
                    category_id=0,
                    category_name='car'
                )
            )
            coco.add_image(coco_image)
    save_json(data=coco.json, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset_dir',
        default='data',
        help='Dataset directory with videos and annotations'
    )
    parser.add_argument(
        '--coco_json_dest',
        default='coco_json.json',
        help='Destination directory for coco json'
    )
    _args = parser.parse_args()
    main(_args.dataset_dir, _args.coco_json_dest)
