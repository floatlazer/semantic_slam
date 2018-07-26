#ifndef SEMANTIC_OCTREE_H
#define SEMANTIC_OCTREE_H


#include <iostream>
#include <octomap/ColorOcTree.h>
#include <semantics_octree/semantics_octree_node.h>

namespace octomap
{

  /// Tree definition
  template<class SEMANTICS>
  class SemanticsOcTree: public OccupancyOcTreeBase<SemanticsOcTreeNode<SEMANTICS> >
  {
  public:

    /// Default constructor, sets resolution of leafs
    SemanticsOcTree(double resolution);

    /// virtual constructor: creates a new object of same type
    /// (Covariant return type requires an up-to-date compiler)
    SemanticsOcTree* create() const {return new SemanticsOcTree(this->resolution);}

    std::string getTreeType() const {return "ColorOcTree";} // For display in rviz

     /**
     * Prunes a node when it is collapsible. This overloaded
     * version only considers the node occupancy and color for pruning,
     * The confidence is average of children's confidence after pruning.
     * \return true if pruning was successful
     */
    virtual bool pruneNode(SemanticsOcTreeNode<SEMANTICS>* node);

    virtual bool isNodeCollapsible(const SemanticsOcTreeNode<SEMANTICS>* node) const;

    bool isUseSemanticColor(){return this->root->use_semantic_color;}

    void setUseSemanticColor(bool use);

    // set node color at given key or coordinate. Replaces previous color.
    SemanticsOcTreeNode<SEMANTICS>* setNodeColor(const OcTreeKey& key, uint8_t r,
                                 uint8_t g, uint8_t b);

    SemanticsOcTreeNode<SEMANTICS>* setNodeColor(float x, float y,
                                 float z, uint8_t r,
                                 uint8_t g, uint8_t b) {
      OcTreeKey key;
      if (!this->coordToKeyChecked(point3d(x,y,z), key)) return NULL;
      return setNodeColor(key,r,g,b);
    }

    /// Integrate color measurement (RGB) at given key. Average with previous color
    SemanticsOcTreeNode<SEMANTICS>* averageNodeColor(const OcTreeKey& key, uint8_t r,
                                  uint8_t g, uint8_t b);

    /// Integrate color measurement (RGB) at given coordinate. Average with previous color
    SemanticsOcTreeNode<SEMANTICS>* averageNodeColor(float x, float y,
                                      float z, uint8_t r,
                                      uint8_t g, uint8_t b) {
      OcTreeKey key;
      if (!this->coordToKeyChecked(point3d(x,y,z), key)) return NULL;
      return averageNodeColor(key,r,g,b);
    }

    /// Update semantics from a new observation by doing bayesian fusion
    SemanticsOcTreeNode<SEMANTICS>* updateNodeSemantics(const OcTreeKey& key, SEMANTICS obs);

    /// Update semantics from a new observation by doing bayesian fusion
    SemanticsOcTreeNode<SEMANTICS>* updateNodeSemantics(float x, float y, float z, SEMANTICS obs) {
      OcTreeKey key;
      if (!this->coordToKeyChecked(point3d(x,y,z), key)) return NULL;
      return updateNodeSemantics(key, obs);
    }

    // Update inner nodes' occupancy, RGB color and semantics
    void updateInnerOccupancy();

  protected:
    void updateInnerOccupancyRecurs(SemanticsOcTreeNode<SEMANTICS>* node, unsigned int depth);

    /**
     * Static member object which ensures that this OcTree's prototype
     * ends up in the classIDMapping only once. You need this as a
     * static member in any derived octree class in order to read .ot
     * files through the AbstractOcTree factory. You should also call
     * ensureLinking() once from the constructor.
     */
    class StaticMemberInitializer{
       public:
         StaticMemberInitializer() {
           SemanticsOcTree* tree = new SemanticsOcTree(0.1);
           tree->clearKeyRays();
           AbstractOcTree::registerTreeType(tree);
         }

         /**
         * Dummy function to ensure that MSVC does not drop the
         * StaticMemberInitializer, causing this tree failing to register.
         * Needs to be called from the constructor of this octree.
         */
         void ensureLinking() {};
    };

    /// static member to ensure static initialization (only once)
    static StaticMemberInitializer semanticsOcTreeMemberInit;
  };

} // end namespace

// Implementation
#include <semantics_octree/semantics_octree.hxx>
#endif
