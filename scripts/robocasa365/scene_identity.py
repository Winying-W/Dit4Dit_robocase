"""Strict MJCF comparison allowing an explicit default OBJ mesh MIME type.

MuJoCo may serialize content_type="model/obj" after loading a .obj mesh. The
same mesh file already implies that type. Keep every other field and the XML
tree structure in the comparison; numeric state checks remain separate.
"""
import xml.etree.ElementTree as ET


def compare_scene_xml(reference, generated):
    def signature(xml):
        ignored = 0

        def visit(element):
            nonlocal ignored
            attributes = dict(element.attrib)
            if (element.tag == 'mesh' and attributes.get('file', '').endswith('.obj')
                    and attributes.get('content_type') == 'model/obj'):
                del attributes['content_type']
                ignored += 1
            return (element.tag, tuple(sorted(attributes.items())),
                    (element.text or '').strip(), (element.tail or '').strip(),
                    tuple(visit(child) for child in element))

        value = visit(ET.fromstring(xml))
        return value, ignored

    left, left_ignored = signature(reference)
    right, right_ignored = signature(generated)
    return dict(exact_text_match=reference == generated, equivalent=left == right,
                ignored_reference_default_obj_types=left_ignored,
                ignored_generated_default_obj_types=right_ignored,
                rule='Only explicit model/obj content_type on .obj meshes is redundant; '
                     'all other attributes, text, and tree structure must match.')
