/**
* \author Xuan Zhang
* \data Mai-July 2018
*/
#ifndef SEMANTIC_OCTREE_NODE_H
#define SEMANTIC_OCTREE_NODE_H
#include <semantics_octree/semantics_octree.h>
namespace octomap
{

  // Forward declaraton for "friend"
  template <class SEMANTICS> class SemanticsOcTree;

  /// Node definition
  template <class SEMANTICS>
  class SemanticsOcTreeNode : public ColorOcTreeNode {
  public:
    friend class SemanticsOcTree<SEMANTICS>; // Needs access to node children (inherited)

  public:

    /// Default constructor
    SemanticsOcTreeNode() : ColorOcTreeNode(), semantics(), use_semantic_color(true){}

    /// Copy constructor
    SemanticsOcTreeNode(const SemanticsOcTreeNode& rhs)
    {
      copyData(rhs);
    }

    /// Operator
    inline bool operator==(const SemanticsOcTreeNode& rhs) const{
      return (rhs.value == value && rhs.semantics == semantics);
    }

    /// Copy data
    void copyData(const SemanticsOcTreeNode& from)
    {
      ColorOcTreeNode::copyData(from);
      semantics = from.getSemantics();
    }

    /// Get semantics
    inline SEMANTICS getSemantics() const {return semantics;}

    /// Set semantics
    inline void setSemantics(SEMANTICS from){semantics = from;}

    /// Is semantics set: not set if colors are all zeros
    inline bool isSemanticsSet() const;

    /// Update semantics (colors and confidences) from children by doing semantic fusion (using method in template class)
    void updateSemanticsChildren();

    /// Do semantic fusion for children nodes (using method in template class)
    SEMANTICS getFusedChildSemantics() const;

    /// Read from file
    std::istream& readData(std::istream &s);

    /// Write to file, also used to serialize octomap, we change the displayed color here
    std::ostream& writeData(std::ostream &s) const;

  protected:
    SEMANTICS semantics;
    bool use_semantic_color; ///<Whether use semantic color rather than rgb color
  };


  // Node implementation  --------------------------------------
  template<class SEMANTICS>
  bool SemanticsOcTreeNode<SEMANTICS>::isSemanticsSet() const
  {
    return this->semantics.isSemanticsSet();
  }

  template<class SEMANTICS>
  void SemanticsOcTreeNode<SEMANTICS>::updateSemanticsChildren()
  {
    semantics = getFusedChildSemantics();
  }

  template<class SEMANTICS>
  SEMANTICS SemanticsOcTreeNode<SEMANTICS>::getFusedChildSemantics() const
  {
    // Fuse semantics of children node by semantic fusion
    SEMANTICS sem;
    bool fusion_started = false;
    if(children != NULL)
    {
      for(int i = 0; i < 8; i++)
      {
        SemanticsOcTreeNode* child =  static_cast<SemanticsOcTreeNode*>(children[i]);
        if(child != NULL && child->isSemanticsSet())
        {
          if(fusion_started)
            sem = SEMANTICS::semanticFusion(sem, child->getSemantics());
          else
          {
            sem = child->getSemantics();
            fusion_started = true;
          }
        }
      }
    }
    return sem;
  }

  template<class SEMANTICS>
  std::istream& SemanticsOcTreeNode<SEMANTICS>::readData(std::istream &s) {
    s.read((char*) &value, sizeof(value)); // occupancy
    s.read((char*) &color, sizeof(Color)); // color
    return s;
  }

  template<class SEMANTICS>
  std::ostream& SemanticsOcTreeNode<SEMANTICS>::writeData(std::ostream &s) const {
    //TODO adapt to show semantic colors
    s.write((const char*) &value, sizeof(value)); // occupancy
    if(use_semantic_color)
    {
      Color sem_color = semantics.getSemanticColor();
      s.write((const char*) &sem_color, sizeof(Color)); // semantic color
    }
    else
      s.write((const char*) &color, sizeof(Color)); // color
    return s;
  }
} // namespace octomap
#endif // SEMANTIC_OCTREE_NODE_H
