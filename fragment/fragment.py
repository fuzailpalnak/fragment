from typing import Any

import numpy as np
from dataclasses import dataclass


@dataclass()
class Fragment:
    position: Any

    def get_fragment_data(self, image: np.ndarray) -> np.ndarray:
        """
        GET THE FRAGMENT PORTION OF THE DATA FROM THE IMAGE

        :param image: numpy array
        :return:
        """
        return image[
            :,
            self.position[0][0] : self.position[0][1],
            self.position[1][0] : self.position[1][1],
            :,
        ]

    def transfer_fragment(self, transfer_from: np.ndarray, transfer_to: np.ndarray):
        """
        TRANSFER THE FRAGMENT PORTION TO BIGGER IMAGE

        :param transfer_from: data to transfer from transfer [np.array]
        :param transfer_to:  data to transfer to [np.array]
        :return: transferred data [np.array]
        """
        assert transfer_to <= transfer_from, (
            "Expected transfer_to greater than or equal to transfer_from, "
            "Expected >= %s got shape %s." % (transfer_to, transfer_from)
        )
        part_1_x = self.position[0][0]
        part_1_y = self.position[0][1]
        part_2_x = self.position[1][0]
        part_2_y = self.position[1][1]

        cropped_image = transfer_to[:, part_1_x:part_1_y, part_2_x:part_2_y, :]

        merged = cropped_image + transfer_from

        if np.any(cropped_image):
            intersecting_elements = np.zeros(cropped_image.shape)
            intersecting_elements[cropped_image > 0] = 1

            non_intersecting_elements = 1 - intersecting_elements

            intersected_with_merged = merged * intersecting_elements
            aggregate_merged = intersected_with_merged / 2

            non_intersected_with_merged = np.multiply(non_intersecting_elements, merged)
            merged = aggregate_merged + non_intersected_with_merged

        transfer_to[:, part_1_x:part_1_y, part_2_x:part_2_y, :] = merged
        return transfer_to


class ImageFragment:
    def __init__(self, fragment_size, image_size):
        self.fragment_size = fragment_size
        self.image_size = image_size
        self.collection = self.fragments()

    def __len__(self):
        return len(self.collection)

    def __getitem__(self, index):
        return self.collection[index]

    def fragments(self):
        """
        W = Columns
        H = Rows
        Input = W x H
        Provides collection of Fragments of fragment_size over org_size, The functions will yield non overlapping
        fragments if img_size / split_size is divisible, if that's not the case then the function will adjust
        the fragments accordingly to accommodate the split_size and yield overlapping fragments
        :return:
        """
        fragment_list = list()

        split_col, split_row = (self.fragment_size[0], self.fragment_size[1])

        img_col = self.image_size[0]
        img_row = self.image_size[1]

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
                    fragment_list.append(
                        Fragment(((row, row + split_row), (col, col + split_col)))
                    )
            iter_row = 1
        return fragment_list

    @classmethod
    def get_image_fragment(cls, fragment_size: tuple, org_size: tuple):
        """
        :param fragment_size:
        :param org_size:
        :return:
        """
        assert len(fragment_size) == 3, (
            "Expected fragment to have shape (width, height, [channels]), "
            "got shape %s." % (fragment_size,)
        )

        assert len(org_size) == 3, (
            "Expected Image to have shape (width, height, [channels]), "
            "got shape %s." % (org_size,)
        )
        assert org_size >= fragment_size, (
            "Expected org_size greater than or equal to fragment_size, "
            "Expected >= %s got shape %s." % (fragment_size, org_size)
        )

        return ImageFragment(fragment_size, org_size)
