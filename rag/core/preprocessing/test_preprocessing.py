#!/usr/bin/env python3
"""
Test VulRAG Preprocessing - Minimal
===================================

Test basique de l'implémentation minimaliste
"""
def test_basic():
    try:
        from rag.core.preprocessing.preprocessing import create_pipeline
        
        # Test code with a buffer overflow vulnerability
        test_code = """
        #include <string.h>
        void vulnerable_function(char* input) {
            char buffer[10];
            strcpy(buffer, input);  
        }
        """
        
        print("="*80)
        print("STARTING PREPROCESSING PIPELINE TEST")
        print("="*80)
        
        print("\n[1/3] Creating pipeline...")
        pipeline = create_pipeline()
        
        print("\n[2/3] Processing code...")
        result = pipeline.process(test_code)
        
        # Print detailed results
        print("\n" + "="*80)
        print("PROCESSING RESULTS")
        print("="*80)
        print(f"\nPurpose:\n{result.purpose}")
        print(f"\nFunction:\n{result.function}")
        print(f"\nEmbedding shape: {result.graph_embedding.shape}")
        print(f"First 10 embedding values: {result.graph_embedding[:10]}")
        print(f"\nProcessing time: {result.processing_time_ms:.1f}ms")
        print(f"Cache hit: {result.cache_hit}")
        
        # Show query dictionary for fusion_controller
        query_dict = result.to_query_dict()
        print("\n" + "="*80)
        print("FUSION CONTROLLER QUERY DICT")
        print("="*80)
        for key, value in query_dict.items():
            if key == 'kb2_vector':
                print(f"{key}: [vector of length {len(value)}]")
                print(f"First 5 values: {value[:5]}...")
            else:
                print(f"\n{key}:")
                print("-"*40)
                print(str(value)[:500] + ("..." if len(str(value)) > 500 else ""))
        
        print("\n" + "="*80)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*80)
        
    except Exception as e:
        print("\n" + "!"*80)
        print("TEST FAILED")
        print("!"*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic()