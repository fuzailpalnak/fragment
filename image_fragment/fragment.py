import math
from typing import Tuple

import numpy as np


class Fragment:
    def __init__(self, position):
        self._position = position

    @property
    def position(self):
        return self._position

    @property
    def height(self):
        return int(math.ceil((self.position[0][1] - self.position[0][0])))

    @property
    def width(self):
        return int(math.ceil((self.position[1][1] - self.position[1][0])))

    def get_fragment_data(self, image: np.ndarray) -> np.ndarray:
        """
        GET THE FRAGMENT PORTION OF THE DATA FROM THE IMAGE

        :param image: numpy array
        :return:
        """
        raise NotImplementedError

    def transfer_fragment(
        self, transfer_from: np.ndarray, transfer_to: np.ndarray
    ) -> np.ndarray:
        """
        TRANSFER THE FRAGMENT PORTION TO BIGGER IMAGE
        :param transfer_from: data to transfer from transfer [np.array]
        :param transfer_to:  data to transfer to [np.array]
        :return: transferred data [np.array]
        """
        raise NotImplementedError

    @staticmethod
    def solve_overlap(cropped_image, merged):
        intersecting_elements = np.zeros(cropped_image.shape)
        intersecting_elements[cropped_image > 0] = 1

        non_intersecting_elements = 1 - intersecting_elements

        intersected_with_merged = merged * intersecting_elements
        aggregate_merged = intersected_with_merged / 2

        non_intersected_with_merged = np.multiply(non_intersecting_elements, merged)
        merged = aggregate_merged + non_intersected_with_merged
        return merged


class Fragment4D(Fragment):
    def __init__(self, position):
        super().__init__(position)

    def get_fragment_data(self, image: np.ndarray) -> np.ndarray:
        """
        GET THE FRAGMENT PORTION OF THE DATA FROM THE IMAGE

        :param image: numpy array
        :return:
        """
        assert len(image.shape) == 4, (
            "Expected fragment to have shape (batch, height, width, [channels]), "
            "got shape %s." % (image.shape,)
        )

        _, w, h, _ = image.shape

        assert (
            w >= self.width and h >= self.height
        ), "Expected image to have [Height] and [Width] greater than fragment, " "Expected %s got shape %s." % (
            (self.height, self.width),
            (w, h),
        )
        return image[
            :,
            self.position[0][0] : self.position[0][1],
            self.position[1][0] : self.position[1][1],
            :,
        ]

    def transfer_fragment(self, transfer_from: np.ndarray, transfer_to: np.ndarray):
        """
        TRANSFER THE FRAGMENT PORTION TO BIGGER IMAGE
        shape 4->[BATCH x H x W x C]
        :param transfer_from: data to transfer from transfer [np.array]
        :param transfer_to:  data to transfer to [np.array]
        :return: transferred data [np.array]
        """
        assert len(transfer_from.shape) == 4, (
            "Expected fragment to have shape (batch, height, width, [channels]), "
            "got shape %s." % (transfer_from.shape,)
        )

        assert len(transfer_to.shape) == 4, (
            "Expected Image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (transfer_to.shape,)
        )

        assert transfer_to.shape >= transfer_from.shape, (
            "Expected transfer_to greater than or equal to transfer_from, "
            "Expected >= %s got shape %s." % (transfer_to, transfer_from)
        )

        _, h_tt, w_tt, _ = transfer_to.shape
        _, h_tf, w_tf, _ = transfer_from.shape
        assert (
            h_tt >= self.height and w_tt >= self.width
        ), "Expected transfer_to to have [Height] and [Width] greater than fragment, " "Expected %s got shape %s." % (
            (self.height, self.width),
            (h_tt, w_tt),
        )

        assert (
            h_tf >= self.height and w_tf >= self.width
        ), "Expected transfer_from to have [Height] and [Width] equal to fragment, " "Expected %s got shape %s." % (
            (self.height, self.width),
            (h_tf, w_tf),
        )

        part_1_x = self.position[0][0]
        part_1_y = self.position[0][1]
        part_2_x = self.position[1][0]
        part_2_y = self.position[1][1]

        cropped_image = transfer_to[:, part_1_x:part_1_y, part_2_x:part_2_y, :]
        merged = cropped_image + transfer_from

        if np.any(cropped_image):
            merged = self.solve_overlap(cropped_image, merged)
        transfer_to[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = merged
        return transfer_to


class Fragment3D(Fragment):
    def __init__(self, position):
        super().__init__(position)

    def get_fragment_data(self, image: np.ndarray) -> np.ndarray:
        """
        GET THE FRAGMENT PORTION OF THE DATA FROM THE IMAGE

        :param image: numpy array
        :return:
        """
        assert len(image.shape) == 3, (
            "Expected fragment to have shape (height, width, [channels]), "
            "got shape %s." % (image.shape,)
        )

        w, h, _ = image.shape

        assert (
            w >= self.width and h >= self.height
        ), "Expected image to have [Height] and [Width] greater than fragment, " "Expected %s got shape %s." % (
            (self.height, self.width),
            (w, h),
        )
        return image[
            self.position[0][0] : self.position[0][1],
            self.position[1][0] : self.position[1][1],
            :,
        ]

    def transfer_fragment(self, transfer_from: np.ndarray, transfer_to: np.ndarray):
        """
        TRANSFER THE FRAGMENT PORTION TO BIGGER IMAGE
        shape 3->[H x W x C]
        :param transfer_from: data to transfer from transfer [np.array]
        :param transfer_to:  data to transfer to [np.array]
        :return: transferred data [np.array]
        """
        assert len(transfer_from.shape) == 3, (
            "Expected fragment to have shape (height, width, [channels]), "
            "got shape %s." % (transfer_from.shape,)
        )

        assert len(transfer_to.shape) == 3, (
            "Expected Image to have shape (height, width, [channels]), "
            "got shape %s." % (transfer_to.shape,)
        )

        assert transfer_to.shape >= transfer_from.shape, (
            "Expected transfer_to greater than or equal to transfer_from, "
            "Expected >= %s got shape %s." % (transfer_to, transfer_from)
        )

        h_tt, w_tt, _ = transfer_to.shape
        h_tf, w_tf, _ = transfer_from.shape
        assert (
            h_tt >= self.height and w_tt >= self.width
        ), "Expected transfer_to to have [Height] and [Width] greater than fragment, " "Expected %s got shape %s." % (
            (self.height, self.width),
            (h_tt, w_tt),
        )

        assert (
            h_tf >= self.height and w_tf >= self.width
        ), "Expected transfer_from to have [Height] and [Width] equal to fragment, " "Expected %s got shape %s." % (
            (self.height, self.width),
            (h_tf, w_tf),
        )

        part_1_x = self.position[0][0]
        part_1_y = self.position[0][1]
        part_2_x = self.position[1][0]
        part_2_y = self.position[1][1]

        cropped_image = transfer_to[part_1_x:part_1_y, part_2_x:part_2_y, :]

        merged = cropped_image + transfer_from

        if np.any(cropped_image):
            merged = self.solve_overlap(cropped_image, merged)
        transfer_to[part_1_x:part_1_y, part_2_x:part_2_y, :] = merged
        return transfer_to


class ImageFragment:
    def __init__(self, collection):
        self.collection = collection

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        return self.collection[index]

    @staticmethod
    def fragments(section_dim: Tuple[int, int], img_dim: Tuple[int, int]):
        """

        W = Columns
        H = Rows
        Provides collection of Fragments of fragment_size over org_size, The functions will yield non overlapping
        fragments if img_size / split_size is divisible, if that's not the case then the function will adjust
        the fragments accordingly to accommodate the split_size and yield overlapping fragments
        :return:
        """

        split_col, split_row = (section_dim[1], section_dim[0])
        img_col, img_row = (img_dim[1], img_dim[0])

        iter_col = 1
        iter_row = 1

        for col in range(0, img_col, split_col):
            if iter_col == np.ceil(img_col / split_col):
                col = img_col - split_col
            else:
                iter_col += 1
            for row in range(0, img_row, split_row):
                if iter_row == np.ceil(img_row / split_row):
                    row = img_row - split_row
                else:
                    iter_row += 1
                if row + split_row <= img_row and col + split_col <= img_col:
                    yield (row, row + split_row), (col, col + split_col)
            iter_row = 1

    @classmethod
    def image_fragment_4d(
        cls,
        fragment_size: Tuple[int, int, int, int],
        org_size: Tuple[int, int, int, int],
    ):
        """
        fragment size = [batch_size, height, width, channel]
        org_size = [batch_size, height, width, channel]

        :param fragment_size:
        :param org_size:
        :return:
        """
        assert len(fragment_size) == 4, (
            "Expected fragment to have shape (batch, height, width, [channels]), "
            "got shape %s." % (fragment_size,)
        )

        assert len(org_size) == 4, (
            "Expected Image to have shape (batch, height, width, [channels]), "
            "got shape %s." % (org_size,)
        )
        assert org_size >= fragment_size, (
            "Expected org_size greater than or equal to fragment_size, "
            "Expected >= %s got shape %s." % (fragment_size, org_size)
        )

        fragment_list = list()
        for position in cls.fragments(
            (fragment_size[1], fragment_size[2]), (org_size[1], org_size[2])
        ):
            fragment_list.append(Fragment4D(position=position))
        return ImageFragment(fragment_list)

    @classmethod
    def image_fragment_3d(
        cls, fragment_size: Tuple[int, int, int], org_size: Tuple[int, int, int]
    ):
        """
        fragment size = [height, width, channel]
        org_size = [height, width, channel]

        :param fragment_size:
        :param org_size:
        :return:
        """
        assert len(fragment_size) == 3, (
            "Expected fragment to have shape (height, width, [channels]), "
            "got shape %s." % (fragment_size,)
        )

        assert len(org_size) == 3, (
            "Expected Image to have shape (height, width, [channels]), "
            "got shape %s." % (org_size,)
        )
        assert org_size >= fragment_size, (
            "Expected org_size greater than or equal to fragment_size, "
            "Expected >= %s got shape %s." % (fragment_size, org_size)
        )
        fragment_list = list()
        for position in cls.fragments(
            (fragment_size[0], fragment_size[1]), (org_size[0], org_size[1])
        ):
            fragment_list.append(Fragment3D(position=position))
        return ImageFragment(fragment_list)
