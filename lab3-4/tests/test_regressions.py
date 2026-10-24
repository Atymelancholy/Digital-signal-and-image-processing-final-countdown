import unittest
import numpy as np
from src.lab2 import kmeans
from src.lab3 import bilinear_sample, rotate_around, upright_angle, _open_double_arc_templates, open_arc_needs_flip
from src.lab4 import guess_figure
from src.lab4_data import resize_gray, augment_images


class GeometryTests(unittest.TestCase):
    def test_zero_rotation_preserves_edges_and_dtype(self):
        for a in (np.arange(25, dtype=np.uint8).reshape(5, 5),
                  np.linspace(0, 1, 25, dtype=np.float32).reshape(5, 5),
                  np.ones((1, 1), dtype=bool), np.full((3, 4, 3), 500, dtype=np.uint16)):
            out = rotate_around(a, 0, 0, 0)
            np.testing.assert_array_equal(a, out)
            self.assertEqual(a.dtype, out.dtype)

    def test_half_turn_preserves_border(self):
        a = np.arange(25, dtype=np.uint8).reshape(5, 5)
        np.testing.assert_array_equal(rotate_around(a, 180, 2, 2), a[::-1, ::-1])

    def test_fractional_boundary_uses_background_per_neighbour(self):
        a = np.full((2, 2), 10.)
        result = bilinear_sample(a, np.array([1., 1.5, 2.]), np.array([1., 1., 1.]))
        np.testing.assert_allclose(result, [10, 5, 0])

    def test_single_pixel_resize(self):
        np.testing.assert_allclose(resize_gray(np.ones((1, 1)), 8, 8), 1)

    def test_constant_resize_preserves_border(self):
        np.testing.assert_allclose(resize_gray(np.ones((3, 5)), 13, 19), 1)

    def test_open_arcs_face_left_after_alignment(self):
        template = _open_double_arc_templates()[1]
        self.assertFalse(open_arc_needs_flip(template))
        self.assertTrue(open_arc_needs_flip(template[::-1, ::-1]))
        image = np.pad(template, ((20, 20), (28, 28)))
        for angle in (0, 35, 90, 180, -50):
            rotated = rotate_around(image, angle, 43.5, 43.5)
            result = rotate_around(rotated, upright_angle(rotated), 43.5, 43.5)
            self.assertFalse(open_arc_needs_flip(result), angle)

    def test_blank_orientation(self):
        self.assertEqual(upright_angle(np.zeros((8, 8), dtype=bool)), 0)


class ClusteringTests(unittest.TestCase):
    def test_identical_points(self):
        np.testing.assert_array_equal(kmeans(np.ones((4, 5)), 3), np.zeros(4))

    def test_fewer_unique_points_than_clusters(self):
        x = np.array([[0., 0.], [0., 0.], [10., 10.], [10., 10.]])
        labels = kmeans(x, 3)
        self.assertEqual(labels[0], labels[1])
        self.assertEqual(labels[2], labels[3])
        self.assertNotEqual(labels[0], labels[2])

    def test_empty_and_single(self):
        self.assertEqual(kmeans(np.empty((0, 2)), 3).shape, (0,))
        np.testing.assert_array_equal(kmeans(np.ones((1, 2)), 3), [0])


class RecognitionTests(unittest.TestCase):
    def test_displayed_prediction_and_confidence_are_network_outputs(self):
        class FixedModel:
            def predict_proba(self, x):
                return np.tile(np.array([.1, 0, 0, 0, 0, 0, 0, 0, .9, 0], dtype=np.float32), (len(x), 1))
        mask = np.zeros((30, 30), bool)
        mask[3:27, 3:27] = True
        mask[7:23, 7:23] = False  # Ноль не должен принудительно заменять ответ 8.
        rgb = np.repeat((mask * 255).astype(np.uint8)[..., None], 3, axis=2)
        result = guess_figure(FixedModel(), rgb, mask, 1)
        self.assertEqual(result.pred, 8)
        self.assertAlmostEqual(result.conf, .9)
        np.testing.assert_array_equal(result.probs, FixedModel().predict_proba([0])[0])

    def test_augmentation_no_wraparound(self):
        x = np.zeros((8, 28, 28), np.float32)
        x[:, 0, 0] = 1
        y = augment_images(x, np.random.default_rng(0))
        self.assertTrue((y[:, -3:, :] == 0).all())
        self.assertTrue((y[:, :, -3:] == 0).all())
        self.assertTrue(((y >= 0) & (y <= 1)).all())


if __name__ == '__main__':
    unittest.main()
