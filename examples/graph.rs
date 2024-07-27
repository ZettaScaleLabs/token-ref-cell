use std::{
    collections::{hash_map::Entry, HashMap},
    fmt::Debug,
    mem,
    sync::Arc,
};

use token_cell2::{token::BoxedToken, Ref, TokenCell};

#[derive(Debug, Default)]
pub struct Graph {
    nodes: HashMap<NodeId, Arc<NodeCell>>,
    token: BoxedToken,
}

pub type NodeId = usize;

type NodeCell = TokenCell<Node, BoxedToken>;

#[derive(Debug, Default)]
pub struct Node {
    id: NodeId,
    edges: HashMap<NodeId, Arc<NodeCell>>,
}

pub struct NodeRef<'a>(Ref<'a, Node, BoxedToken>);

impl<'a> Clone for NodeRef<'a> {
    fn clone(&self) -> Self {
        Self(Ref::clone(&self.0))
    }
}

impl<'a> NodeRef<'a> {
    pub fn id(&self) -> NodeId {
        self.0.id
    }

    pub fn edges(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.0.edges.keys().copied()
    }
}

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_node(&self, id: NodeId) -> Option<NodeRef> {
        Some(NodeRef(self.nodes.get(&id)?.borrow(&self.token)))
    }

    pub fn insert_node(&mut self, id: NodeId) -> bool {
        let entry = self.nodes.entry(id);
        let inserted = matches!(entry, Entry::Vacant(_));
        entry.or_insert_with(|| Node::new_cell(id, &self.token));
        inserted
    }

    pub fn remove_node(&mut self, id: NodeId) -> bool {
        let Some(node) = self.nodes.remove(&id) else {
            return false;
        };
        for _ in node.borrow_mut(&mut self.token).reborrow_iter_mut(
            |node| node.edges.values().map(AsRef::as_ref),
            |mut other| other.edges.remove(&id),
        ) {}
        true
    }

    pub fn insert_edge(&mut self, node1: NodeId, node2: NodeId) {
        let mut node = |id| {
            let entry = self.nodes.entry(id);
            let node = entry.or_insert_with(|| Node::new_cell(node1, &self.token));
            node.clone()
        };
        let node_cell1 = node(node1);
        let node_cell2 = node(node2);
        node_cell1
            .borrow_mut(&mut self.token)
            .edges
            .insert(node2, node_cell2.clone());
        node_cell2
            .borrow_mut(&mut self.token)
            .edges
            .insert(node1, node_cell1);
    }

    pub fn remove_edge(&mut self, node1: NodeId, node2: NodeId) {
        let mut remove = |a, b| {
            let Some(node) = self.nodes.get(&a) else {
                return false;
            };
            node.borrow_mut(&mut self.token).edges.remove(&b);
            true
        };
        if remove(node1, node2) {
            remove(node2, node1);
        }
    }

    pub fn clear(&mut self) {
        let mut nodes = mem::take(&mut self.nodes);
        for (_, node) in nodes.drain() {
            if let Ok(mut node) = node.try_borrow_mut(&mut self.token) {
                node.edges.clear();
            }
        }
        self.nodes = nodes;
    }
}

impl Node {
    fn new_cell(id: NodeId, token: &BoxedToken) -> Arc<NodeCell> {
        let edges = HashMap::new();
        Arc::new(NodeCell::new(Node { id, edges }, token))
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        self.clear()
    }
}

fn main() {
    let mut graph = Graph::new();
    graph.insert_edge(0, 1);
    graph.insert_edge(1, 2);
    graph.insert_edge(2, 0);
    graph.remove_node(1);
    let graph = Arc::new(graph);
    let node = graph.get_node(0).unwrap();
    assert_eq!(node.id(), 0);
    let mut edges = node.edges();
    assert_eq!(edges.next(), Some(2));
    assert_eq!(edges.next(), None);
}
