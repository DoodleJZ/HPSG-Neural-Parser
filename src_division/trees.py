import collections.abc
import gzip
Sub_Head = "<S>"
Head = "<H>"
No_Head = "<N>"
Htype = 1
Ntype = 0
class TreebankNode(object):
    pass

class InternalTreebankNode(TreebankNode):
    def __init__(self, label, children):
        assert isinstance(label, str)
        self.label = label
        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, TreebankNode) for child in children)
        assert children
        self.children = tuple(children)

        self.father = self.children[0].father
        self.type = self.children[0].type
        self.head = self.children[0].head

        self.left = self.children[0].left
        self.right = self.children[-1].right
        self.cun = 0
        for child in self.children:
            if child.father < self.left + 1 or child.father > self.right:
                self.father = child.father
                self.type = child.type
                self.head = child.head
                self.cun += 1

    def linearize(self):
        return "({} {})".format(
            self.label, " ".join(child.linearize() for child in self.children))

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def convert(self, index=0, node_type = Htype, nocache=False):
        tree = self

        sublabels = []
        if node_type == Htype:
            sublabels.append(Head)

        sublabels.append(self.label)

        while len(tree.children) == 1 and isinstance(
                tree.children[0], InternalTreebankNode):
            tree = tree.children[0]
            sublabels.append(tree.label)

        children = []
        sub_father = set()
        sub_head = set()
        al_make = set()
        if len(tree.children) == 1:
            children.append(tree.children[0].convert(index=index))
            return InternalParseNode(tuple(sublabels), children, nocache=nocache)

        for child in tree.children:
            sub_head |= set([child.head])
            sub_father |= set([child.father])

        flag = 0
        #clear multi-head
        for child in tree.children:
            # in sub tree
            if (child.father in sub_head and child.father != self.head) or (
                    child.head in sub_father and child.head != self.head):
                sub_r = child.father
                if child.head in sub_father:
                    sub_r = child.head
                if sub_r not in al_make:
                    al_make |= set([sub_r])
                else:
                    continue
                sub_children = []
                sub_flag = 0
                for sub_child in tree.children:
                    if sub_child.father == sub_r or sub_child.head == sub_r:
                        if sub_flag == 0:
                            if isinstance(sub_child, InternalTreebankNode):
                                sub_node = sub_child.convert(index=index, node_type=Htype)
                            else:
                                # head leaf need to assign label
                                sub_node = InternalParseNode(tuple([Head]), [sub_child.convert(index=index)], nocache=nocache)
                        else:
                            if isinstance(sub_child, InternalTreebankNode):
                                sub_node = sub_child.convert(index=index, node_type=Ntype)
                            else:
                                sub_node = sub_child.convert(index=index)
                        if len(sub_children) > 0:
                            assert sub_children[-1].right == sub_node.left  # contiune span
                        sub_children.append(sub_node)

                        if sub_child.head == sub_r:
                            sub_flag = 1
                        index = sub_children[-1].right

                assert len(sub_children) > 1

                if flag == 0:
                    sub_node = InternalParseNode(tuple([Head] + [Sub_Head]), sub_children, nocache=nocache)
                else :
                    sub_node = InternalParseNode(tuple([Sub_Head]), sub_children, nocache=nocache)

                children.append(sub_node)
            else:
                if len(children) > 0:
                    assert children[-1].right == child.left
                if flag == 0:
                    if isinstance(child, InternalTreebankNode):
                        children.append(child.convert(index=index, node_type=Htype))
                    else:
                        # head leaf need to assign label
                        children.append(
                            InternalParseNode(tuple([Head]), [child.convert(index=index)], nocache=nocache))
                else:
                    if isinstance(child, InternalTreebankNode):
                        children.append(child.convert(index=index, node_type=Ntype))
                    else:
                        # right of the head
                        children.append(child.convert(index=index))

            if children[-1].head == self.head:
                flag = 1
            index = children[-1].right

        return InternalParseNode(tuple(sublabels), children, nocache=nocache)

class LeafTreebankNode(TreebankNode):
    def __init__(self, tag, word, head, father, type):
        assert isinstance(tag, str)
        self.tag = tag
        self.father = father
        self.type = type
        self.head = head
        assert isinstance(word, str)
        self.word = word
        self.left = self.head - 1
        self.right = self.head

    def linearize(self):
        return "({} {})".format(self.tag, self.word)

    def leaves(self):
        yield self

    def convert(self, index=0):
        return LeafParseNode(index, self.tag, self.word, self.type, self.father )

class ParseNode(object):
    pass

class InternalParseNode(ParseNode):
    def __init__(self, label, children, maketree = 0,  nodetype = None, nocache=False):
        assert isinstance(label, tuple)
        assert all(isinstance(sublabel, str) for sublabel in label)
        assert label
        self.label = label

        self.node_type = nodetype
        if self.node_type is None:
            if self.label[0] == Head:
                self.node_type = Htype
            else:
                self.node_type = Ntype

        assert isinstance(children, collections.abc.Sequence)
        assert all(isinstance(child, ParseNode) for child in children)
        assert children
        assert len(children) > 1 or isinstance(children[0], LeafParseNode)
        assert all(
            left.right == right.left
            for left, right in zip(children, children[1:]))
        self.children = tuple(children)

        self.left = children[0].left
        self.right = children[-1].right
        self.type = type
        self.father = 0
        self.head = None
        if maketree ==0 :
            for child in self.children:
                if child.father - 1 < self.left or child.father > self.right:
                    self.type = child.type
                    self.father = child.father
                    self.head = child.head
        else:
            self.head = children[0].head
            for child in self.children:
                if child.node_type == Htype:
                    self.head = child.head
            for child in self.children:
                if self.head != child.head:
                    child.make_fa(self.head)

        self.cun_w = 0
        for child in self.children:
            if self.father != child.father:
                if child.father != self.head:
                    self.cun_w += 1

        self.nocache = nocache

    def leaves(self):
        for child in self.children:
            yield from child.leaves()

    def make_fa(self, father):
        self.father = father
        for child in self.children:
            if child.head == self.head:
                child.make_fa(father)

    def convert(self):
        children = []
        for child in self.children:
            child_node = child.convert()
            if isinstance(child_node, list):
                children = children + child_node
            else :
                children.append(child_node)
        if self.label[0] == Sub_Head:
            return  children
        tree = InternalTreebankNode(self.label[-1], children)
        for sublabel in reversed(self.label[:-1]):
            tree = InternalTreebankNode(sublabel, [tree])
        return tree


    def enclosing(self, left, right):
        assert self.left <= left < right <= self.right
        for child in self.children:
            if isinstance(child, LeafParseNode):
                continue
            if child.left <= left < right <= child.right:
                return child.enclosing(left, right)
        return self

    def oracle_label(self, left, right):
        enclosing = self.enclosing(left, right)
        if enclosing.left == left and enclosing.right == right:
            return enclosing.label
        return ()

    def oracle_splits(self, left, right):
        return [
            child.left
            for child in self.enclosing(left, right).children
            if left < child.left < right
        ]

class LeafParseNode(ParseNode):
    def __init__(self, index, tag, word, type, father = 0, nodetype = None):
        assert isinstance(index, int)
        assert index >= 0
        self.left = index
        self.right = index + 1

        assert isinstance(tag, str)
        self.tag = tag
        self.head = index + 1
        self.node_type = nodetype
        if self.node_type is None:
            self.node_type = Ntype
        self.father = father
        self.type = type

        assert isinstance(word, str)
        self.word = word

    def leaves(self):
        yield self

    def make_fa(self, father):
        self.father = father

    def convert(self):
        return LeafTreebankNode(tag=self.tag, word=self.word, head=self.head, father=self.father, type = self.type)

def load_trees(path, heads = None, types = None, strip_top=True):
    with open(path) as infile:
        treebank = infile.read()

    tokens = treebank.replace("(", " ( ").replace(")", " ) ").split()

    cun_word = 0 #without root
    cun_sent = 0
    def helper(index, flag_sent):
        nonlocal cun_sent
        nonlocal cun_word
        trees = []

        while index < len(tokens) and tokens[index] == "(":
            paren_count = 0
            while tokens[index] == "(":
                index += 1
                paren_count += 1

            label = tokens[index]

            index += 1

            if tokens[index] == "(":
                children, index = helper(index, flag_sent = 0)
                if len(children) > 0:
                    trees.append(InternalTreebankNode(label, children))
            else:
                word = tokens[index]
                index += 1
                if label != '-NONE-':
                    trees.append(LeafTreebankNode(label, word, father = heads[cun_sent][cun_word], type = types[cun_sent][cun_word], head = cun_word + 1))
                    if cun_sent<0:
                        print(cun_sent, cun_word, word, heads[cun_sent][cun_word], types[cun_sent][cun_word])
                    cun_word += 1

            while paren_count > 0:
                assert tokens[index] == ")"
                index += 1
                paren_count -= 1

            if flag_sent == 1 :
                cun_sent += 1
                cun_word = 0

        return trees, index

    trees, index = helper(0, flag_sent = 1)
    assert index == len(tokens)
    assert len(trees) == cun_sent

    if strip_top:
        for i, tree in enumerate(trees):
            if tree.label in ("TOP", "ROOT"):
                assert len(tree.children) == 1
                trees[i] = tree.children[0]

    def process_NONE(tree):

        if isinstance(tree, LeafTreebankNode):
            label = tree.tag
            if label == '-NONE-':
                return None
            else:
                return tree

        tr = []
        label = tree.label
        if label == '-NONE-':
            return None
        for node in tree.children:
            new_node = process_NONE(node)
            if new_node is not None:
                tr.append(new_node)
        if tr == []:
            return None
        else:
            return InternalTreebankNode(label, tr)

    new_trees = []
    for i, tree in enumerate(trees):
        new_tree = process_NONE(tree)
        new_trees.append(new_tree)

    return new_trees

