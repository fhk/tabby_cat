#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "graph_assembler.hpp"

namespace py = pybind11;

PYBIND11_MODULE(graph_assembler_cpp, m) {
    py::class_<Node>(m, "Node")
        .def_readwrite("id", &Node::id)
        .def_readwrite("x", &Node::x)
        .def_readwrite("y", &Node::y)
        .def_readwrite("prize", &Node::prize);

    py::class_<Edge>(m, "Edge")
        .def_readwrite("u", &Edge::u)
        .def_readwrite("v", &Edge::v)
        .def_readwrite("length", &Edge::length);

    py::class_<GraphAssembler>(m, "GraphAssembler")
        .def(py::init<>())
        .def("load_osm_pbf", &GraphAssembler::load_osm_pbf)
        .def("load_demand_points", &GraphAssembler::load_demand_points)
        .def("snap_demand_to_graph", &GraphAssembler::snap_demand_to_graph)
        .def("get_nodes", &GraphAssembler::get_nodes)
        .def("get_edges", &GraphAssembler::get_edges);
}
