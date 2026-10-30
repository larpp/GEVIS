"""Select the actual training branch for isolated synthetic cell tests."""
import ast


def training_source(source):
    tree=ast.parse(source)
    if (tree.body and isinstance(tree.body[0],ast.If) and
        ast.unparse(tree.body[0].test) in ('not REUSE_SAVED_BACKBONE','not REUSE_SAVED_ENCODERS')):
        return ast.unparse(ast.Module(tree.body[0].body,type_ignores=[]))
    return source
