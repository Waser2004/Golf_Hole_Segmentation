import ast
import copy
import csv
import sys
from math import comb, floor, sqrt
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import numpy as np
from PIL import Image

from golf_hole_segmentation.utils.paths import DATASET_CSV, GOLF_HOLES_DIR, REFERENCE_IMAGES_DIR


class DataConverter:
    def __init__(
        self,
        grid_size: tuple[int, int],
        box_size: float,
        dataset_csv: str | Path = DATASET_CSV,
        reference_images_dir: str | Path = REFERENCE_IMAGES_DIR,
        output_images_dir: str | Path = GOLF_HOLES_DIR,
    ):
        self.GRID_SIZE = grid_size
        self.BOX_SIZE = box_size
        self.dataset_csv = Path(dataset_csv)
        self.reference_images_dir = Path(reference_images_dir)
        self.output_images_dir = Path(output_images_dir)

        self.colors = [
            (50, 205, 50),
            (104, 155, 64),
            (33, 153, 50),
            (20, 101, 33),
            (17, 76, 25),
            (210, 180, 140),
            (240, 230, 140),
            (17, 48, 25),
            (70, 130, 180),
            (255, 255, 255),
            (128, 128, 128),
            (226, 114, 91),
        ]

        self.data = self._load_dataset_rows()
        self.outlines = [[] for _ in self.data]
        self.reference_imgs = [path.name for path in self.reference_images_dir.glob("*.png")]
        self.color_array = np.zeros((self.GRID_SIZE[1], self.GRID_SIZE[0]), dtype=np.int32)

        self.outline_curve_points = []
        self.outline_curve_data = []
        self.offset = 35
        self.converted_data = {}

    def _load_dataset_rows(self):
        with self.dataset_csv.open("r", newline="") as csv_file:
            reader = csv.reader(csv_file)
            rows = [row for row in reader if row]

        return [
            [
                row[0],
                float(row[1]),
                ast.literal_eval(row[2]),
                ast.literal_eval(row[3]),
                ast.literal_eval(row[4]),
                ast.literal_eval(row[5]),
                ast.literal_eval(row[6]),
                ast.literal_eval(row[7]),
            ]
            for row in rows
        ]

    @staticmethod
    def filter_indices_by_par(indices, desired_par, csv_filename="output.csv"):
        index_to_par = {}
        with open(csv_filename, newline="") as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                index_to_par[row[0].strip()] = int(row[1].strip())

        return [idx for idx in indices if idx in index_to_par and index_to_par[idx] == desired_par]

    def convert_all(self, only_par: int | None = None):
        print("Convert all real holes:")
        print("[{}] {}%".format("." * 20, 0), end="", flush=True)

        total = max(len(self.data) - 1, 1)
        for index, data in enumerate(self.data):
            if only_par is None or self._matches_par(data[0], only_par):
                self.convert(data=data)

            progress = index / total
            print("\r", end="")
            print(
                "[{}{}] {}%".format(
                    "=" * floor(progress * 20),
                    "." * (20 - floor(progress * 20)),
                    progress * 100,
                ),
                end="",
                flush=True,
            )

        print("\r", end="")
        print("[----- Complete -----] 100%\n", end="", flush=True)
        print(f"{len(list(self.converted_data.values()))} real holes converted!")
        return list(self.converted_data.values())

    def _matches_par(self, hole_index: str, desired_par: int) -> bool:
        for file_name in self.reference_imgs:
            parts = Path(file_name).stem.split("-")
            if hole_index in file_name and len(parts) >= 3 and int(parts[2]) == desired_par:
                return True
        return False

    def convert(self, data=None, index: int | None = None):
        if data is None and index is None:
            raise ValueError("No data specified. Set an index or pass a data row.")

        self.color_array *= 0

        if data is None:
            data = self.data[index]
        else:
            index = self.data.index(data)

        hole_index = data[0]
        if f"{hole_index}" in self.converted_data:
            return self.converted_data[f"{hole_index}"]

        for outline_data in data[4]:
            self.outlines[index].append(self.calc_outline(outline_data))

        min_x = min(pos[0] for outline in self.outlines[index] for pos in outline)
        max_x = max(pos[0] for outline in self.outlines[index] for pos in outline)
        min_y = min(pos[1] for outline in self.outlines[index] for pos in outline)
        max_y = max(pos[1] for outline in self.outlines[index] for pos in outline)

        delta_x = -((max_x - min_x) / 2 + min_x)
        delta_y = -((max_y - min_y) / 2 + min_y)
        pasted_image_corners = self.generate_segmentation_input(data, delta_x, delta_y)

        for i, outline in enumerate(self.outlines[index]):
            self.outlines[index][i] = [[pos[0] + delta_x, pos[1] + delta_y] for pos in outline]

            for p_index, pos in enumerate(self.outlines[index][i]):
                self.outlines[index][i][p_index][0] *= data[1] / self.BOX_SIZE
                self.outlines[index][i][p_index][1] *= data[1] / self.BOX_SIZE
                self.outlines[index][i][p_index][0] += self.GRID_SIZE[0] / 2
                self.outlines[index][i][p_index][1] += self.GRID_SIZE[1] / 2

            points = np.array(self.outlines[index][i], dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(self.color_array, [points], color=data[5][i] + 1)

        points = np.array(self.calc_hole_outline(data, delta_x, delta_y), dtype=np.int32).reshape(
            (-1, 1, 2)
        )
        mask = np.zeros_like(self.color_array)
        cv2.fillPoly(mask, [points], color=1)
        self.color_array = np.where(mask == 1, self.color_array, 255)

        full_mask = np.ones_like(self.color_array, dtype=np.uint8)
        cv2.fillConvexPoly(full_mask, np.array(pasted_image_corners, dtype=np.int32), color=0)
        self.color_array = np.where(full_mask == 0, self.color_array, 255)

        self.converted_data[f"{hole_index}"] = copy.deepcopy(self.color_array)
        return self.converted_data[f"{hole_index}"]

    def generate_segmentation_input(self, data, hole_delta_x, hole_delta_y):
        self.output_images_dir.mkdir(parents=True, exist_ok=True)
        image_name = next(img for img in self.reference_imgs if data[0] in img)
        ref_image = Image.open(self.reference_images_dir / image_name)

        scale = data[1]
        new_size = (int(ref_image.width * scale), int(ref_image.height * scale))
        scaled_image = ref_image.resize(new_size)

        white_image = Image.new("RGB", self.GRID_SIZE, color="white")
        offset_x = int(hole_delta_x * scale)
        offset_y = int(hole_delta_y * scale)
        x = (self.GRID_SIZE[0] - new_size[0]) // 2 + offset_x
        y = (self.GRID_SIZE[1] - new_size[1]) // 2 + offset_y
        white_image.paste(scaled_image, (x, y))
        white_image.save(self.output_images_dir / f"{data[0]}.png")

        return [(x, y), (x + new_size[0], y), (x + new_size[0], y + new_size[1]), (x, y + new_size[1])]

    def calc_outline(self, data):
        if len(data) < 3:
            return []

        points = [data[-1]] + data + data[0:3]
        polygon_data = []

        for i in range(len(points) - 4):
            vector_points = [points[i + 1][0]]
            if not points[i][1]:
                vector_points.append(
                    self.__get_vector_points(points[i][0], points[i + 1][0], points[i + 2][0])[1]
                )
            if not points[i + 1][1]:
                vector_points.append(
                    self.__get_vector_points(points[i + 1][0], points[i + 2][0], points[i + 3][0])[
                        0
                    ]
                )
            vector_points.append(points[i + 2][0])
            polygon_data.extend(self.__calc_bezier(vector_points))

        return polygon_data

    def calc_hole_outline(self, data, hole_delta_x, hole_delta_y):
        self.outline_curve_points = [[0, 0]] + [[0, 0] for _ in range(len(data[6]) * 2)] + [[0, 0]]

        for i, p in enumerate(data[6]):
            if i == 0:
                delta_x, delta_y = data[6][i + 1][0] - p[0], data[6][i + 1][1] - p[1]
                distance = sqrt(delta_x**2 + delta_y**2)

                dx = -(delta_y * self.offset * data[7][i] / data[1]) / distance
                dy = (delta_x * -dx) / delta_y if delta_y != 0 else self.offset * data[7][i] / data[1]
                self.outline_curve_points[0] = [p[0] - dx, p[1] - dy]
                self.outline_curve_points[2] = [p[0] + dx, p[1] + dy]

                dx = -(delta_x * self.offset * data[7][i] / data[1]) / distance
                dy = -(delta_y * -dx) / delta_x if delta_x != 0 else self.offset * data[7][i] / data[1]
                self.outline_curve_points[1] = [p[0] + dx, p[1] + dy]

            elif i != len(data[6]) - 1:
                delta_x = data[6][i + 1][0] - data[6][i - 1][0]
                delta_y = data[6][i + 1][1] - data[6][i - 1][1]
                distance = sqrt(delta_x**2 + delta_y**2)

                dx = -(delta_y * self.offset * data[7][i] / data[1]) / distance
                dy = (delta_x * -dx) / delta_y if delta_y != 0 else self.offset * data[7][i] / data[1]
                self.outline_curve_points[i + 2] = [p[0] + dx, p[1] + dy]
                self.outline_curve_points[len(self.outline_curve_points) - i] = [p[0] - dx, p[1] - dy]

            else:
                delta_x, delta_y = p[0] - data[6][i - 1][0], p[1] - data[6][i - 1][1]
                distance = sqrt(delta_x**2 + delta_y**2)

                dx = -(delta_y * self.offset * data[7][i] / data[1]) / distance
                dy = (delta_x * -dx) / delta_y if delta_y != 0 else self.offset * data[7][i] / data[1]
                self.outline_curve_points[len(data[6]) + 1] = [p[0] + dx, p[1] + dy]
                self.outline_curve_points[len(data[6]) + 3] = [p[0] - dx, p[1] - dy]

                dx = -(delta_x * self.offset * data[7][i] / data[1]) / distance
                dy = -(delta_y * -dx) / delta_x if delta_x != 0 else self.offset * data[7][i] / data[1]
                self.outline_curve_points[len(data[6]) + 2] = [p[0] - dx, p[1] - dy]

        for p in self.outline_curve_points:
            p[0] = (p[0] + hole_delta_x) * (data[1] / self.BOX_SIZE) + (self.GRID_SIZE[0] / 2)
            p[1] = (p[1] + hole_delta_y) * (data[1] / self.BOX_SIZE) + (self.GRID_SIZE[1] / 2)

        return self.__calc_hole_bezier()

    def __calc_hole_bezier(self):
        points = [self.outline_curve_points[-1]] + self.outline_curve_points + self.outline_curve_points[0:3]
        outline_points = []

        for i in range(len(points) - 4):
            vector_points = [
                points[i + 1],
                self.__get_vector_points(points[i], points[i + 1], points[i + 2])[1],
                self.__get_vector_points(points[i + 1], points[i + 2], points[i + 3])[0],
                points[i + 2],
            ]
            outline_points.extend(self.__calc_bezier(vector_points))

        return outline_points

    @staticmethod
    def __get_vector_points(p1, p2, p3):
        dx, dy = p3[0] - p1[0], p3[1] - p1[1]
        p13_dis = sqrt((p3[0] - p1[0]) ** 2 + (p3[1] - p1[1]) ** 2)
        p12_dis = sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
        p23_dis = sqrt((p3[0] - p2[0]) ** 2 + (p3[1] - p2[1]) ** 2)

        if abs(dx) >= abs(dy):
            if dx != 0:
                return (
                    p2[0] - p13_dis / dx * (p12_dis / 3),
                    p2[1] - p13_dis / dx * (p12_dis / 3) * (dy / dx),
                ), (
                    p2[0] + p13_dis / dx * (p23_dis / 3),
                    p2[1] + p13_dis / dx * (p23_dis / 3) * (dy / dx),
                )
            return (p2[0], p2[1] - p13_dis * (p12_dis / 3)), (
                p2[0],
                p2[1] + p13_dis * (p23_dis / 3),
            )

        if dy != 0:
            return (
                p2[0] - p13_dis / dy * (p12_dis / 3) * (dx / dy),
                p2[1] - p13_dis / dy * (p12_dis / 3),
            ), (
                p2[0] + p13_dis / dy * (p23_dis / 3) * (dx / dy),
                p2[1] + p13_dis / dy * (p23_dis / 3),
            )
        return (p2[0] - p13_dis * (p12_dis / 3), p2[1]), (
            p2[0] + p13_dis * (p23_dis / 3),
            p2[1],
        )

    @staticmethod
    def __calc_bezier(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
        res = round(sqrt((points[-1][0] - points[0][0]) ** 2 + (points[-1][1] - points[0][1]) ** 2))
        n = len(points) - 1
        curve_points = []
        for i in range(res):
            t = i / (res - 1) if res - 1 > 0 else 1
            point = [0, 0]
            for j in range(n + 1):
                cof = comb(n, j) * t**j * (1 - t) ** (n - j)
                point[0] += cof * points[j][0]
                point[1] += cof * points[j][1]
            curve_points.append(tuple(point))
        return curve_points


Data_converter = DataConverter
