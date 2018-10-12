#include <boost/python.hpp>
#include "pointcloudpair.h"

namespace bp = boost::python;

bool rmbp(size_t N, bp::list &compatible_edge, bp::list &incompatible_edge, bp::list &result)
{
  std::vector<std::pair<size_t, size_t> > positive_edge, negative_edge;
  for(int i = 0; i < bp::len(compatible_edge); ++i)
  {
    bp::list e = boost::python::extract<bp::list>(compatible_edge[i]);
    std::pair<size_t, size_t> pair(boost::python::extract<int>(e[0]), boost::python::extract<int>(e[1]));
    positive_edge.push_back(pair);
  }

  for(int i = 0; i < bp::len(incompatible_edge); ++i)
  {
    bp::list e = boost::python::extract<bp::list>(incompatible_edge[i]);
    std::pair<size_t, size_t> pair(boost::python::extract<int>(e[0]), boost::python::extract<int>(e[1]));
    negative_edge.push_back(pair);
  }

  double belief_threshold = 0.6;
  size_t max_iteration = 1000;
  PointCloudPair pc_pair(N);
  pc_pair.SetBeliefThreshold(belief_threshold);
  pc_pair.SetMaxIteration(max_iteration);
  std::vector<std::pair<size_t, size_t>> refine_match_pairs;
  if (!pc_pair.BeliefPropagation_withPredefinedGraph(positive_edge, negative_edge, refine_match_pairs))
  {
    return false;
  }
  for(size_t i=0; i<refine_match_pairs.size(); ++i)
  {
    result.append(refine_match_pairs[i]);
  }
  return true;
}

BOOST_PYTHON_MODULE(rmbp)
{
    using namespace boost::python;
    Py_Initialize();
    def("rmbp", rmbp);
}
