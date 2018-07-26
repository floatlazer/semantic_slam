namespace octomap {

  // Tree implementation  --------------------------------------
  template<class SEMANTICS>
  SemanticsOcTree<SEMANTICS>::SemanticsOcTree(double resolution)
  : OccupancyOcTreeBase<SemanticsOcTreeNode<SEMANTICS> >(this->resolution)
  {
    semanticsOcTreeMemberInit.ensureLinking();
  };

  template<class SEMANTICS>
  bool SemanticsOcTree<SEMANTICS>::pruneNode(SemanticsOcTreeNode<SEMANTICS>* node) {
    // Same as ColorOcTree
    if (!isNodeCollapsible(node))
      return false;

    // Set value to children's values (all assumed equal)
    node->copyData(*(this->getNodeChild(node, 0)));

    if (node->isColorSet()) // TODO check
      node->setColor(node->getAverageChildColor());

    // Delete children
    for (unsigned int i=0;i<8;i++) {
      this->deleteNodeChild(node, i);
    }
    delete[] node->children;
    node->children = NULL;

    return true;
  }

  template<class SEMANTICS>
  bool SemanticsOcTree<SEMANTICS>::isNodeCollapsible(const SemanticsOcTreeNode<SEMANTICS>* node) const
  {
    // All children must exist, must not have children of
    // their own and have same occupancy probability and same semantic
    if(!this->nodeChildExists(node, 0))
      return false;
    const SemanticsOcTreeNode<SEMANTICS>* firstChild = this->getNodeChild(node, 0);
    if(this->nodeHasChildren(firstChild))
      return false;
    for (unsigned int i = 1; i<8; i++) {
      // Compare nodes using their occupancy and semantics, ignoring color for pruning
      if (!this->nodeChildExists(node, i) || this->nodeHasChildren(this->getNodeChild(node, i))
          || !(this->getNodeChild(node, i)->getValue() == firstChild->getValue())
          || !(this->getNodeChild(node, i)->getSemantics() == firstChild->getSemantics()))
        return false;
    }
    return true;
  }

  template<class SEMANTICS>
  void SemanticsOcTree<SEMANTICS>::setUseSemanticColor(bool use)
  {
    // Traverse all tree nodes
    for(typename SemanticsOcTree<SEMANTICS>::tree_iterator it = this->begin_tree(), end=this->end_tree(); it!= end; ++it)
      it->use_semantic_color = use;
  }

  template<class SEMANTICS>
  SemanticsOcTreeNode<SEMANTICS>* SemanticsOcTree<SEMANTICS>::setNodeColor(const OcTreeKey& key,
                                             uint8_t r,
                                             uint8_t g,
                                             uint8_t b)
  {
    SemanticsOcTreeNode<SEMANTICS>* n = this->search (key);
    if (n != 0) {
      n->setColor(r, g, b);
    }
    return n;
  }

  template<class SEMANTICS>
  SemanticsOcTreeNode<SEMANTICS>* SemanticsOcTree<SEMANTICS>::averageNodeColor(const OcTreeKey& key, uint8_t r,
                                uint8_t g, uint8_t b)
  {
    SemanticsOcTreeNode<SEMANTICS>* n = this->search(key);
    if (n != 0) {
      if (n->isColorSet()) {
        ColorOcTreeNode::Color prev_color = n->getColor();
        n->setColor((prev_color.r + r)/2, (prev_color.g + g)/2, (prev_color.b + b)/2);
      }
      else {
        n->setColor(r, g, b);
      }
    }
    return n;
  }

  template<class SEMANTICS>
  SemanticsOcTreeNode<SEMANTICS>* SemanticsOcTree<SEMANTICS>::updateNodeSemantics(const OcTreeKey& key, SEMANTICS obs)
  {
    SemanticsOcTreeNode<SEMANTICS>* n = this->search(key);
    if (n != 0)
    {
      if (n->isSemanticsSet())
      {
        SEMANTICS sem = SEMANTICS::semanticFusion(n->semantics, obs);
        n->setSemantics(sem);
      }
      else
      {
        n->setSemantics(obs);
      }
    }
    return n;
  }

  template<class SEMANTICS>
  void SemanticsOcTree<SEMANTICS>::updateInnerOccupancy() {
    this->updateInnerOccupancyRecurs(this->root, 0);
  }

  template<class SEMANTICS>
  void SemanticsOcTree<SEMANTICS>::updateInnerOccupancyRecurs(SemanticsOcTreeNode<SEMANTICS>* node, unsigned int depth) {
    // Only recurse and update for inner nodes:
    if (this->nodeHasChildren(node)){
      // Return early for last level:
      if (depth < this->tree_depth){
        for (unsigned int i=0; i<8; i++) {
          if (this->nodeChildExists(node, i)) {
            updateInnerOccupancyRecurs(this->getNodeChild(node, i), depth+1);
          }
        }
      }
      // Update occupancy, color and semantics for inner nodes
      node->updateOccupancyChildren();
      node->updateColorChildren();
      node->updateSemanticsChildren();
    }
  }

  template<class SEMANTICS>
  typename SemanticsOcTree<SEMANTICS>::StaticMemberInitializer SemanticsOcTree<SEMANTICS>::semanticsOcTreeMemberInit;

} // end namespace
