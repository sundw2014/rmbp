#include <boost/python.hpp>
#include "pointcloudpair.h"

namespace bp = boost::python;

bool rmbp(size_t N, bp::list &compatible_edge, bp::list &incompatible_edge, bp::list &result)
{
  std::tr1::unordered_map<size_t, std::vector<size_t> >  positive_match_map, negative_match_map;
  for(int i = 0; i < bp::len(compatible_edge); ++i)
  {
    bp::list v = boost::python::extract<bp::list>(compatible_edge[i]);
    std::vector<size_t> pairs;
    for(int j = 0; j < bp::len(v); ++j)
    {
      pairs.push_back(boost::python::extract<int>(v[j]));
    }
    positive_match_map[i] = pairs;
  }

  for(int i = 0; i < bp::len(incompatible_edge); ++i)
  {
    bp::list v = boost::python::extract<bp::list>(incompatible_edge[i]);
    std::vector<size_t> pairs;
    for (int j = 0; j < bp::len(v); ++j)
    {
      pairs.push_back(boost::python::extract<int>(v[j]));
    }
    negative_match_map[i] = pairs;
  }
  double belief_threshold = 0.6;
  size_t max_iteration = 1000;
  PointCloudPair pc_pair(N);
  pc_pair.SetBeliefThreshold(belief_threshold);
  pc_pair.SetMaxIteration(max_iteration);
  std::vector<std::pair<size_t, size_t>> refine_match_pairs;
  if (!pc_pair.BeliefPropagation_withPredefinedGraph(refine_match_pairs))
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
