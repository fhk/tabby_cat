#include "graph_assembler.hpp"
#include <osmium/io/any_input.hpp>
#include <osmium/handler.hpp>
#include <osmium/visitor.hpp>
#include <osmium/index/map/flex_mem.hpp>
#include <osmium/handler/node_locations_for_ways.hpp>
#include <osmium/geom/haversine.hpp>

#include <ogrsf_frmts.h>
#include <ogr_geometry.h>

#include <cmath>
#include <algorithm>
#include <iostream>
#include <spatialindex/SpatialIndex.h>

using index_type = osmium::index::map::FlexMem<osmium::unsigned_object_id_type, osmium::Location>;
using location_handler_type = osmium::handler::NodeLocationsForWays<index_type>;

class OSMHandler : public osmium::handler::Handler {
    GraphAssembler& assembler_;
public:
    OSMHandler(GraphAssembler& assembler) : assembler_(assembler) {}

    void way(const osmium::Way& way) {
        const char* highway = way.tags().get_value_by_key("highway");
        if (highway) {
            static const std::vector<std::string> valid_highways = {
                "residential", "primary", "secondary", "tertiary",
                "service", "unclassified", "trunk", "motorway",
                "motorway_link"
            };

            bool valid = false;
            for (const auto& h : valid_highways) {
                if (h == highway) {
                    valid = true;
                    break;
                }
            }

            if (valid && way.nodes().size() >= 2) {
                for (size_t i = 0; i < way.nodes().size() - 1; ++i) {
                    const auto& n1 = way.nodes()[i];
                    const auto& n2 = way.nodes()[i+1];

                    if (n1.location() && n2.location()) {
                        double dist = osmium::geom::haversine::distance(n1.location(), n2.location());
                        assembler_.add_edge_from_osm(n1.location().lon(), n1.location().lat(),
                                                   n2.location().lon(), n2.location().lat(), dist);
                    }
                }
            }
        }
    }
};

namespace {
    double distance_pt_seg(double px, double py, double x1, double y1, double x2, double y2, double& outX, double& outY) {
        double dx = x2 - x1;
        double dy = y2 - y1;
        if (dx == 0 && dy == 0) {
            outX = x1;
            outY = y1;
            return std::sqrt((px-x1)*(px-x1) + (py-y1)*(py-y1));
        }
        double t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy);
        t = std::max(0.0, std::min(1.0, t));
        outX = x1 + t * dx;
        outY = y1 + t * dy;
        return std::sqrt((px-outX)*(px-outX) + (py-outY)*(py-outY));
    }
}

GraphAssembler::GraphAssembler() {
    GDALAllRegister();
}

GraphAssembler::~GraphAssembler() {}

void GraphAssembler::load_osm_pbf(const std::string& filepath) {
    index_type index;
    location_handler_type location_handler{index};
    OSMHandler data_handler{*this};

    osmium::io::Reader reader{filepath};
    osmium::apply(reader, location_handler, data_handler);
    reader.close();
}

void GraphAssembler::load_demand_points(const std::string& filepath) {
    GDALDataset *poDS = (GDALDataset*) GDALOpenEx(filepath.c_str(), GDAL_OF_VECTOR, NULL, NULL, NULL);
    if (poDS == NULL) return;

    OGRLayer *poLayer = poDS->GetLayer(0);
    if (poLayer == NULL) {
        GDALClose(poDS);
        return;
    }

    poLayer->ResetReading();
    OGRFeature *poFeature;
    while ((poFeature = poLayer->GetNextFeature()) != NULL) {
        OGRGeometry *poGeometry = poFeature->GetGeometryRef();
        if (poGeometry != NULL && wkbFlatten(poGeometry->getGeometryType()) == wkbPoint) {
            OGRPoint *poPoint = (OGRPoint *) poGeometry;
            Node n;
            n.id = next_node_id_++;
            n.x = poPoint->getX();
            n.y = poPoint->getY();

            // Try to get prize from attributes
            int prize_idx = poFeature->GetFieldIndex("prize");
            if (prize_idx >= 0) {
                n.prize = poFeature->GetFieldAsDouble(prize_idx);
            } else {
                n.prize = 1.0;
            }
            demand_points_.push_back(n);
        }
        OGRFeature::DestroyFeature(poFeature);
    }
    GDALClose(poDS);
}

long long GraphAssembler::get_or_create_node(double x, double y) {
    auto it = coord_to_id_.find({x, y});
    if (it != coord_to_id_.end()) {
        return it->second;
    }
    long long id = next_node_id_++;
    nodes_[id] = {id, x, y, 0.0};
    coord_to_id_[{x, y}] = id;
    return id;
}

void GraphAssembler::add_edge_from_osm(double x1, double y1, double x2, double y2, double dist) {
    long long u = get_or_create_node(x1, y1);
    long long v = get_or_create_node(x2, y2);
    edges_.push_back({u, v, dist});
}

void GraphAssembler::snap_demand_to_graph() {
    if (edges_.empty() || demand_points_.empty()) return;

    using namespace SpatialIndex;
    IStorageManager* storage = StorageManager::createNewMemoryStorageManager();
    id_type indexIdentifier;
    ISpatialIndex* tree = RTree::createNewRTree(*storage, 0.7, 100, 100, 2, RTree::RV_RSTAR, indexIdentifier);

    for (size_t i = 0; i < edges_.size(); ++i) {
        const auto& edge = edges_[i];
        const auto& n1 = nodes_[edge.u];
        const auto& n2 = nodes_[edge.v];
        double low[] = {std::min(n1.x, n2.x), std::min(n1.y, n2.y)};
        double high[] = {std::max(n1.x, n2.x), std::max(n1.y, n2.y)};
        Region r(low, high, 2);
        tree->insertData(0, nullptr, r, static_cast<id_type>(i));
    }

    std::vector<Edge> new_edges;
    std::vector<size_t> edges_to_remove;

    for (auto& demand : demand_points_) {
        double pt[] = {demand.x, demand.y};
        Point p(pt, 2);

        struct Visitor : public IVisitor {
            double min_dist = 1e18;
            size_t best_edge_idx = -1;
            double best_x, best_y;
            const Node& demand;
            const std::vector<Edge>& edges;
            std::unordered_map<long long, Node>& nodes;

            Visitor(const Node& d, const std::vector<Edge>& e, std::unordered_map<long long, Node>& n)
                : demand(d), edges(e), nodes(n) {}

            void visitNode(const INode& n) override {}
            void visitData(const IData& d) override {
                size_t edge_idx = static_cast<size_t>(d.getIdentifier());
                const auto& edge = edges[edge_idx];
                const auto& n1 = nodes[edge.u];
                const auto& n2 = nodes[edge.v];
                double snapX, snapY;
                double dist = distance_pt_seg(demand.x, demand.y, n1.x, n1.y, n2.x, n2.y, snapX, snapY);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_edge_idx = edge_idx;
                    best_x = snapX;
                    best_y = snapY;
                }
            }
            void visitData(std::vector<const IData*>& v) override {}
        };

        Visitor v(demand, edges_, nodes_);
        // Query nearest neighbors or within a range.
        // For simplicity, let's query a small region around the point first,
        // or just use the R-tree to prune.
        double delta = 0.01; // roughly 1km
        double low[] = {demand.x - delta, demand.y - delta};
        double high[] = {demand.x + delta, demand.y + delta};
        Region r(low, high, 2);
        tree->intersectsWithQuery(r, v);

        if (v.best_edge_idx != (size_t)-1) {
            long long new_id = get_or_create_node(v.best_x, v.best_y);
            nodes_[new_id].prize += demand.prize;

            const auto& old_edge = edges_[v.best_edge_idx];
            if (new_id != old_edge.u && new_id != old_edge.v) {
                double d1 = std::sqrt(std::pow(v.best_x - nodes_[old_edge.u].x, 2) + std::pow(v.best_y - nodes_[old_edge.u].y, 2));
                double d2 = std::sqrt(std::pow(v.best_x - nodes_[old_edge.v].x, 2) + std::pow(v.best_y - nodes_[old_edge.v].y, 2));

                new_edges.push_back({old_edge.u, new_id, d1});
                new_edges.push_back({new_id, old_edge.v, d2});
                edges_to_remove.push_back(v.best_edge_idx);
            }
        }
    }

    // Sort and remove duplicates from edges_to_remove, then remove from edges_
    std::sort(edges_to_remove.begin(), edges_to_remove.end());
    edges_to_remove.erase(std::unique(edges_to_remove.begin(), edges_to_remove.end()), edges_to_remove.end());

    std::vector<Edge> final_edges;
    std::unordered_set<size_t> to_remove_set(edges_to_remove.begin(), edges_to_remove.end());
    for (size_t i = 0; i < edges_.size(); ++i) {
        if (to_remove_set.find(i) == to_remove_set.end()) {
            final_edges.push_back(edges_[i]);
        }
    }
    final_edges.insert(final_edges.end(), new_edges.begin(), new_edges.end());
    edges_ = std::move(final_edges);

    delete tree;
    delete storage;
}

std::vector<Node> GraphAssembler::get_nodes() const {
    std::vector<Node> res;
    for (const auto& kv : nodes_) {
        res.push_back(kv.second);
    }
    return res;
}

std::vector<Edge> GraphAssembler::get_edges() const {
    return edges_;
}
