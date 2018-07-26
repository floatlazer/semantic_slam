#include<semantics_octree/semantics_max.h>
namespace octomap
{
  std::ostream& operator<<(std::ostream& out, SemanticsMax const& s)
  {
    return out << '(' << s.semantic_color << ", " << s.confidence << ')';
  }
}
