#include "stdafx.h"
#include "CppUnitTest.h"
#include "AllDifferentUtils.h"

using namespace hydra;
using namespace std;
using namespace Microsoft::VisualStudio::CppUnitTestFramework;

namespace HydraSolverTest {
	TEST_CLASS(AllDifferentUtilsTest) {
public:
	TEST_METHOD(FordFulkersonAlgorithmShouldReturnValidFlowValueForEveryNodes_fourNodeGraph) {
		vector<AllDiffNode*> nodes;
		auto node1 = new AllDiffNode(SOURCE);
		auto node2 = new AllDiffNode(VARIABLE);
		auto node3 = new AllDiffNode(VARIABLE);
		auto node4 = new AllDiffNode(SINK);

		auto edge12 = new AllDiffEdge(5, node1, node2);
		auto edge13 = new AllDiffEdge(3, node1, node3);
		node1->adjencyList.push_back(edge12);
		node1->adjencyList.push_back(edge13);

		auto edge23 = new AllDiffEdge(2, node2, node3);
		auto edge24 = new AllDiffEdge(3, node2, node4);
		node2->adjencyList.push_back(edge23);
		node2->adjencyList.push_back(edge24);

		auto edge34 = new AllDiffEdge(5, node3, node4);
		node3->adjencyList.push_back(edge34);

		nodes.push_back(node1);
		nodes.push_back(node2);
		nodes.push_back(node3);
		nodes.push_back(node4);

		FordFulkersonAlgorithm(nodes, node1, node4);

		Assert::AreEqual(5, edge12->flow);
		Assert::AreEqual(3, edge13->flow);
		Assert::AreEqual(2, edge23->flow);
		Assert::AreEqual(5, edge34->flow);
		Assert::AreEqual(3, edge24->flow);

		for (auto node : nodes) {
			for (auto edge : node->adjencyList) {
				delete edge;
			}
			delete node;
		}
	}

	TEST_METHOD(FordFulkersonAlgorithmShouldReturnValidFlowValueForEveryNodes_sixNodeGraph) {
		vector<AllDiffNode*> nodes;
		auto node1 = new AllDiffNode(SOURCE);
		auto node2 = new AllDiffNode(VARIABLE);
		auto node3 = new AllDiffNode(VARIABLE);
		auto node4 = new AllDiffNode(VARIABLE);
		auto node5 = new AllDiffNode(VARIABLE);
		auto node6 = new AllDiffNode(SINK);

		auto edge12 = new AllDiffEdge(3, node1, node2);
		auto edge13 = new AllDiffEdge(7, node1, node3);
		node1->adjencyList.push_back(edge12);
		node1->adjencyList.push_back(edge13);

		auto edge24 = new AllDiffEdge(2, node2, node4);
		auto edge25 = new AllDiffEdge(2, node2, node5);
		node2->adjencyList.push_back(edge24);
		node2->adjencyList.push_back(edge25);

		auto edge35 = new AllDiffEdge(4, node3, node5);
		node3->adjencyList.push_back(edge35);

		auto edge46 = new AllDiffEdge(3, node4, node6);
		node4->adjencyList.push_back(edge46);

		auto edge56 = new AllDiffEdge(5, node5, node6);
		node5->adjencyList.push_back(edge56);

		nodes.push_back(node1);
		nodes.push_back(node2);
		nodes.push_back(node3);
		nodes.push_back(node4);
		nodes.push_back(node5);
		nodes.push_back(node6);

		FordFulkersonAlgorithm(nodes, node1, node6);

		Assert::AreEqual(5, edge56->flow);
		Assert::AreEqual(2, edge46->flow);
		Assert::AreEqual(2, edge24->flow);
		Assert::AreEqual(1, edge25->flow);
		Assert::AreEqual(3, edge12->flow);
		Assert::AreEqual(4, edge13->flow);
		Assert::AreEqual(4, edge35->flow);

		deleteGraph(nodes);
	}

	TEST_METHOD(KosarajuAlgorithmShouldReturnStronglyConnectedComponents_fourNodeGraph) {
		vector<AllDiffNode*> nodes;
		auto node1 = new AllDiffNode(SOURCE);
		auto node2 = new AllDiffNode(VARIABLE);
		auto node3 = new AllDiffNode(VALUE);
		auto node4 = new AllDiffNode(SINK);

		auto edge12 = new AllDiffEdge(1, node1, node2);
		node1->adjencyList.push_back(edge12);
		auto edge23 = new AllDiffEdge(1, node2, node3);
		node2->adjencyList.push_back(edge23);
		auto edge31 = new AllDiffEdge(1, node3, node1);
		auto edge34 = new AllDiffEdge(1, node3, node4);
		node3->adjencyList.push_back(edge31);
		node3->adjencyList.push_back(edge34);

		nodes.push_back(node1);
		nodes.push_back(node2);
		nodes.push_back(node3);
		nodes.push_back(node4);

		auto stronglyConnectedComponents = KosarajuAlgorithm(nodes);

		// There should be 2 sets, one containing node 1, 2 and 3 and another one containing node 4
		size_t expectedSize = 2;
		Assert::AreEqual(expectedSize, stronglyConnectedComponents.size());

		for (auto component : stronglyConnectedComponents) {
			if (component.find(node1) != component.end()) {
				Assert::IsTrue(component.find(node2) != component.end());
				Assert::IsTrue(component.find(node3) != component.end());
				Assert::IsFalse(component.find(node4) != component.end());
			} else {
				Assert::IsTrue(component.find(node4) != component.end());
				Assert::IsFalse(component.find(node1) != component.end());
				Assert::IsFalse(component.find(node2) != component.end());
				Assert::IsFalse(component.find(node3) != component.end());
			}
		}

		deleteGraph(nodes);
	}

	TEST_METHOD(KosarajuAlgorithmShouldReturnStronglyConnectedComponents_eightNodeGraph) {
		vector<AllDiffNode*> nodes;
		auto node1 = new AllDiffNode(SOURCE);
		auto node2 = new AllDiffNode(VARIABLE);
		auto node3 = new AllDiffNode(VALUE);
		auto node4 = new AllDiffNode(SINK);
		auto node5 = new AllDiffNode(SOURCE);
		auto node6 = new AllDiffNode(VARIABLE);
		auto node7 = new AllDiffNode(VALUE);
		auto node8 = new AllDiffNode(SINK);

		auto edge12 = new AllDiffEdge(1, node1, node2);
		node1->adjencyList.push_back(edge12);

		auto edge23 = new AllDiffEdge(1, node2, node3);
		auto edge25 = new AllDiffEdge(1, node2, node5);
		auto edge26 = new AllDiffEdge(1, node2, node6);
		node2->adjencyList.push_back(edge23);
		node2->adjencyList.push_back(edge25);
		node2->adjencyList.push_back(edge26);

		auto edge34 = new AllDiffEdge(1, node3, node4);
		auto edge37 = new AllDiffEdge(1, node3, node7);
		node3->adjencyList.push_back(edge34);
		node3->adjencyList.push_back(edge37);

		auto edge43 = new AllDiffEdge(1, node4, node3);
		auto edge48 = new AllDiffEdge(1, node4, node8);
		node4->adjencyList.push_back(edge43);
		node4->adjencyList.push_back(edge48);

		auto edge51 = new AllDiffEdge(1, node5, node1);
		auto edge56 = new AllDiffEdge(1, node5, node6);
		node5->adjencyList.push_back(edge51);
		node5->adjencyList.push_back(edge56);

		auto edge67 = new AllDiffEdge(1, node6, node7);
		node6->adjencyList.push_back(edge67);

		auto edge76 = new AllDiffEdge(1, node7, node6);
		node7->adjencyList.push_back(edge76);

		auto edge84 = new AllDiffEdge(1, node8, node4);
		auto edge87 = new AllDiffEdge(1, node8, node7);
		node8->adjencyList.push_back(edge84);
		node8->adjencyList.push_back(edge87);

		nodes.push_back(node1);
		nodes.push_back(node2);
		nodes.push_back(node3);
		nodes.push_back(node4);
		nodes.push_back(node5);
		nodes.push_back(node6);
		nodes.push_back(node7);
		nodes.push_back(node8);

		auto stronglyConnectedComponents = KosarajuAlgorithm(nodes);

		// There should be 3 sets, one containing node 1, 2 and 5, one containing 6 and 7 and one containing 3, 4 and 8
		size_t expectedSize = 3;
		Assert::AreEqual(expectedSize, stronglyConnectedComponents.size());

		for (auto component : stronglyConnectedComponents) {
			if (component.find(node1) != component.end()) {
				Assert::IsTrue(component.size() == 3);
				Assert::IsTrue(component.find(node2) != component.end());
				Assert::IsTrue(component.find(node5) != component.end());
			} else if (component.find(node3) != component.end()) {
				Assert::IsTrue(component.size() == 3);
				Assert::IsTrue(component.find(node4) != component.end());
				Assert::IsTrue(component.find(node8) != component.end());
			} else {
				Assert::IsTrue(component.size() == 2);
				Assert::IsTrue(component.find(node6) != component.end());
				Assert::IsTrue(component.find(node7) != component.end());
			}
		}

		deleteGraph(nodes);
	}
	};
}
