use std::{collections::HashMap, mem, sync::Arc};

use token_cell2::{token::BoxedToken, TokenCell};

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

impl Graph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_node(&self, id: NodeId) -> Option<&Node> {
        Some(self.nodes.get(&id)?.borrow(&self.token).inner)
    }

    pub fn insert_node(&mut self, id: NodeId) -> &mut Node {
        let entry = self.nodes.entry(id);
        let node = entry.or_insert_with(|| Node::new_cell(id, &self.token));
        node.borrow_mut(&mut self.token).inner
    }

    pub fn remove_node(&mut self, id: NodeId) -> bool {
        let Some(node) = self.nodes.remove(&id) else {
            return false;
        };
        let node = node.borrow_mut(&mut self.token);
        for other in node.inner.edges.values() {
            other.borrow_mut(&mut *node.token).edges.remove(&id);
        }
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

    pub fn id(&self) -> NodeId {
        self.id
    }

    pub fn edges(&self) -> impl Iterator<Item = NodeId> + '_ {
        self.edges.keys().copied()
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
