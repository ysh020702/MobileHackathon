import unittest
from kitti_metrics import prepare_frame, statistics, score_class


class KittiMetricTests(unittest.TestCase):
    def test_perfect_and_missing_class(self):
        truth = {str(i): [('Car', 0, 0, [0, 0, 100, 100])] for i in range(100)}
        predictions = {i: [[0, 0, 100, 100, .9]] for i in truth}
        self.assertEqual(score_class(truth, predictions, 'Car')['ap40'], 100)
        self.assertEqual(score_class(truth, {}, 'Car')['ap40'], 0)
        self.assertIsNone(score_class(truth, {}, 'Cyclist')['ap40'])

    def test_moderate_boundaries(self):
        rows = [('Car', .3, 1, [0, 0, 100, 26]),
                ('Car', .3, 1, [0, 0, 100, 25]),
                ('Car', .31, 1, [0, 0, 100, 50]),
                ('Car', 0, 2, [0, 0, 100, 50]),
                ('Van', 0, 0, [0, 0, 100, 50])]
        self.assertEqual(prepare_frame(rows, [], 'Car')['gt'].tolist(), [0, 1, 1, 1, 1])

    def test_duplicate_and_dontcare(self):
        rows = [('Car', 0, 0, [0, 0, 100, 100]),
                ('DontCare', -1, -1, [200, 0, 300, 100])]
        predictions = [[0, 0, 100, 100, .9], [0, 0, 100, 100, .8],
                       [200, 0, 300, 100, .7], [400, 0, 410, 20, .6]]
        self.assertEqual(statistics(prepare_frame(rows, predictions, 'Car'), .7, 0)[:3], (1, 1, 0))

    def test_strict_iou_and_neighbor(self):
        rows = [('Car', 0, 0, [0, 0, 100, 100])]
        self.assertEqual(statistics(prepare_frame(rows, [[0, 0, 70, 100, .9]], 'Car'), .7, 0)[:3], (0, 1, 1))
        rows = [('Van', 0, 0, [0, 0, 100, 100])]
        self.assertEqual(statistics(prepare_frame(rows, [[0, 0, 100, 100, .9]], 'Car'), .7, 0)[:3], (0, 0, 0))


if __name__ == '__main__':
    unittest.main()
