import unittest

from scripts.robocasa365.scene_identity import compare_scene_xml


class SceneIdentityTests(unittest.TestCase):
    def test_default_obj_type_is_redundant(self):
        reference = '<mujoco><asset><mesh file="part.obj" content_type="model/obj"/></asset></mujoco>'
        generated = '<mujoco><asset><mesh file="part.obj"/></asset></mujoco>'
        result = compare_scene_xml(reference, generated)
        self.assertTrue(result['equivalent'])
        self.assertFalse(result['exact_text_match'])
        self.assertEqual(result['ignored_reference_default_obj_types'], 1)

    def test_nondefault_type_is_not_ignored(self):
        for filename, mime in [('part.stl', 'model/obj'), ('part.obj', 'model/stl')]:
            with self.subTest(filename=filename, mime=mime):
                reference = f'<mesh file="{filename}" content_type="{mime}"/>'
                generated = f'<mesh file="{filename}"/>'
                self.assertFalse(compare_scene_xml(reference, generated)['equivalent'])

    def test_other_mesh_changes_are_rejected(self):
        reference = '<mesh file="part.obj" content_type="model/obj" scale="1 1 1"/>'
        for generated in ['<mesh file="other.obj" scale="1 1 1"/>',
                          '<mesh file="part.obj" scale="2 1 1"/>']:
            self.assertFalse(compare_scene_xml(reference, generated)['equivalent'])

    def test_physics_changes_are_rejected(self):
        reference = '<mujoco><option timestep="0.002"/><worldbody><body pos="0 0 1"/></worldbody></mujoco>'
        for generated in [reference.replace('0.002', '0.001'), reference.replace('0 0 1', '0 0 2')]:
            self.assertFalse(compare_scene_xml(reference, generated)['equivalent'])

    def test_tree_structure_is_preserved(self):
        reference = '<body name="a"><body name="b"/></body>'
        generated = '<body name="b"><body name="a"/></body>'
        self.assertFalse(compare_scene_xml(reference, generated)['equivalent'])

    def test_attributes_on_other_tags_are_preserved(self):
        reference = '<texture file="part.obj" content_type="model/obj"/>'
        generated = '<texture file="part.obj"/>'
        self.assertFalse(compare_scene_xml(reference, generated)['equivalent'])


if __name__ == '__main__':
    unittest.main()
