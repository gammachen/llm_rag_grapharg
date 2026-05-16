#!/usr/bin/env python3
"""测试中文实体提取格式"""

import ollama

# 使用中文文本测试
test_prompt = """
-Goal-
Given a text document that is potentially relevant to this activity and a list of entity types, identify all entities of those types from the text and all relationships among the identified entities.

-Steps-
1. Identify all entities. For each identified entity, extract the following information:
- entity_name: Name of the entity, capitalized
- entity_type: One of the following types: [person, location, event, organization, concept]
- entity_description: Comprehensive description of the entity's attributes and activities
Format each entity as ("entity"<|><entity_name><|><entity_type><|><entity_description>

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are *clearly related* to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity, as identified in step 1
- target_entity: name of the target entity, as identified in step 1
- relationship_description: explanation as to why you think the source entity and the target entity are related to each other
- relationship_strength: a numeric score indicating strength of the relationship between the source entity and target entity
 Format each relationship as ("relationship"<|><source_entity><|><target_entity><|><relationship_description><|><relationship_strength>)

3. Return output in English as a single list of all the entities and relationships identified in steps 1 and 2. Use **##** as the list delimiter.

4. When finished, output <|COMPLETE|>

######################
-Real Data-
######################
Entity_types: [person, location, event, organization, concept]
Text: 段誉是一位年轻的公子，他来到无量山剑湖宫观看比武。左子穆是无量剑东宗的掌门。龚光杰是左子穆的弟子。
######################
Output:
"""

print("测试中文实体提取...")

try:
    response = ollama.generate(
        model="phi4-mini:3.8b",
        prompt=test_prompt,
        options={'temperature': 0.1, 'num_predict': 1000}
    )
    
    print("\nLLM返回结果:")
    print("-" * 60)
    print(repr(response['response']))
    print("-" * 60)
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
