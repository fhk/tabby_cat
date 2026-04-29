#ifndef GRAPH_ASSEMBLER_HPP
#define GRAPH_ASSEMBLER_HPP

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <unordered_set>

struct Node {
    long long id;
    double x, y;
    double prize;
};

struct Edge {
    long long u, v;
    double length;
};

class GraphAssembler {
public:
    GraphAssembler();
    ~GraphAssembler();

    void load_osm_pbf(const std::string& filepath);
    void load_demand_points(const std::string& filepath);

    // Snapping logic
    void snap_demand_to_graph();

    // Data retrieval
    std::vector<Node> get_nodes() const;
    std::vector<Edge> get_edges() const;

    // Internal use but exposed for the handler
    void add_edge_from_osm(double x1, double y1, double x2, double y2, double dist);

private:
    std::unordered_map<long long, Node> nodes_;
    std::vector<Edge> edges_;
    std::vector<Node> demand_points_;

    long long next_node_id_ = 0;
    std::map<std::pair<double, double>, long long> coord_to_id_;

    long long get_or_create_node(double x, double y);
};

#endif // GRAPH_ASSEMBLER_HPP
