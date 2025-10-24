#!/usr/bin/env python3
"""
RAG Database Population Script
Loads the 2.2M recipe dataset into Weaviate vector database for enhanced RAG system
"""

import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent / "backend"))

from app.services.enhanced_rag_system import EnhancedRAGSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RAGDatabasePopulator:
    """Populates the RAG database with recipe data."""
    
    def __init__(self, data_path: str = "data/recipes.json"):
        self.data_path = Path(data_path)
        self.rag_system = None
        self.batch_size = 100  # Process recipes in batches
        
    async def initialize_rag_system(self):
        """Initialize the RAG system."""
        try:
            self.rag_system = EnhancedRAGSystem(
                weaviate_url="http://localhost:8080",
                recipe_db_path=None,  # We'll populate manually
                model_name="all-MiniLM-L6-v2"
            )
            await self.rag_system.initialize()
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def load_recipe_data(self) -> List[Dict[str, Any]]:
        """Load recipe data from JSON file."""
        try:
            if not self.data_path.exists():
                logger.error(f"Recipe data file not found: {self.data_path}")
                return []
            
            with open(self.data_path, 'r', encoding='utf-8') as f:
                recipes = json.load(f)
            
            logger.info(f"Loaded {len(recipes)} recipes from {self.data_path}")
            return recipes
        
        except Exception as e:
            logger.error(f"Failed to load recipe data: {e}")
            return []
    
    async def process_recipe_batch(self, recipes: List[Dict[str, Any]], batch_num: int):
        """Process a batch of recipes."""
        try:
            logger.info(f"Processing batch {batch_num} ({len(recipes)} recipes)")
            
            for i, recipe in enumerate(recipes):
                try:
                    # Add recipe to RAG system
                    await self.rag_system.add_recipe(recipe)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"  Processed {i + 1}/{len(recipes)} recipes in batch {batch_num}")
                
                except Exception as e:
                    logger.error(f"Failed to process recipe {recipe.get('title', 'Unknown')}: {e}")
                    continue
            
            logger.info(f"Completed batch {batch_num}")
            
        except Exception as e:
            logger.error(f"Failed to process batch {batch_num}: {e}")
    
    async def populate_database(self):
        """Main function to populate the RAG database."""
        try:
            logger.info("Starting RAG database population")
            start_time = time.time()
            
            # Initialize RAG system
            await self.initialize_rag_system()
            
            # Load recipe data
            recipes = self.load_recipe_data()
            if not recipes:
                logger.error("No recipes to process")
                return
            
            # Process recipes in batches
            total_batches = (len(recipes) + self.batch_size - 1) // self.batch_size
            logger.info(f"Processing {len(recipes)} recipes in {total_batches} batches")
            
            for batch_num in range(total_batches):
                start_idx = batch_num * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(recipes))
                batch_recipes = recipes[start_idx:end_idx]
                
                await self.process_recipe_batch(batch_recipes, batch_num + 1)
                
                # Small delay between batches to prevent overwhelming the system
                await asyncio.sleep(0.1)
            
            total_time = time.time() - start_time
            logger.info(f"RAG database population completed in {total_time:.2f} seconds")
            logger.info(f"Processed {len(recipes)} recipes total")
            
        except Exception as e:
            logger.error(f"RAG database population failed: {e}")
            raise
    
    async def verify_population(self):
        """Verify that data was populated correctly."""
        try:
            # Test search functionality
            test_queries = [
                "pasta recipes",
                "chicken dinner",
                "dessert chocolate",
                "vegan protein",
                "quick breakfast"
            ]
            
            logger.info("Verifying RAG database population...")
            
            for query in test_queries:
                results = await self.rag_system.search_recipes(
                    query=query,
                    limit=3,
                    filters={}
                )
                
                logger.info(f"Query '{query}': Found {len(results)} results")
                if results:
                    for i, result in enumerate(results):
                        title = result.get('title', 'Unknown')
                        score = result.get('score', 0)
                        logger.info(f"  {i+1}. {title} (score: {score:.3f})")
            
            logger.info("RAG database verification completed")
            
        except Exception as e:
            logger.error(f"RAG database verification failed: {e}")

async def main():
    """Main function."""
    try:
        # Check if Weaviate is running
        import aiohttp
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get("http://localhost:8080/v1/.well-known/ready") as response:
                    if response.status != 200:
                        logger.error("Weaviate is not ready. Please start Weaviate first.")
                        logger.info("Run: docker run -p 8080:8080 semitechnologies/weaviate:latest")
                        return
            except aiohttp.ClientConnectorError:
                logger.error("Cannot connect to Weaviate. Please start Weaviate first.")
                logger.info("Run: docker run -p 8080:8080 semitechnologies/weaviate:latest")
                return
        
        # Create populator and run
        populator = RAGDatabasePopulator()
        await populator.populate_database()
        await populator.verify_population()
        
        logger.info("✅ RAG database population completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Population interrupted by user")
    except Exception as e:
        logger.error(f"Population failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())