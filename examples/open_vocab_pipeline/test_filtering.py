"""Checks for the RGB-D mask postfilter used before graph construction."""

import unittest

import numpy as np

from run import filter_candidates


def candidate(index, score, mask):
    return {"candidate_index": index, "label": "object", "class_id": 0,
            "confidence": score, "box_xyxy": [0, 0, 20, 20], "mask": mask}


class FilteringTests(unittest.TestCase):
    def test_depth_duplicate_fragment_and_border_rejections(self):
        depth = np.ones((20, 20), dtype=np.float32)
        first = np.zeros((20, 20), dtype=bool)
        first[2:7, 2:7] = True
        second = np.zeros_like(first)
        second[10:15, 2:7] = True
        near_camera = np.zeros_like(first)
        near_camera[10:15, 10:15] = True
        depth[near_camera] = 0.1
        border = np.zeros_like(first)
        border[19, 2:12] = True
        fragmented = np.zeros_like(first)
        fragmented[2:4, 10:12] = True
        fragmented[7:9, 15:17] = True
        proposals = [candidate(0, 0.02, first), candidate(1, 0.019, first.copy()),
                     candidate(2, 0.018, near_camera), candidate(3, 0.017, border),
                     candidate(4, 0.016, np.ones_like(first)),
                     candidate(5, 0.015, fragmented), candidate(6, 0.014, second)]
        kept, filtered = filter_candidates(proposals, depth, 2, 0.15, 2.0, 0.5,
                                           10, 0.5, 0.6, 0.85, False)
        self.assertEqual([item["candidate_index"] for item in kept], [0, 6])
        self.assertEqual([item["mask_index"] for item in kept], [0, 1])
        self.assertEqual({item["reason"] for item in filtered}, {
            "duplicate_mask", "insufficient_valid_depth", "truncated_at_image_border",
            "mask_covers_too_much_of_frame", "fragmented_mask"})
        self.assertTrue(all("mask" not in item for item in filtered))

    def test_rgb_only_mode_keeps_depth_unsupported_mask(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[5:10, 5:10] = True
        kept, filtered = filter_candidates([candidate(0, 0.01, mask)], None, 1,
                                           0.15, 2.0, 0.5, 10, 0.5, 0.6, 0.85, False)
        self.assertEqual(len(kept), 1)
        self.assertEqual(filtered, [])


if __name__ == "__main__":
    unittest.main()
