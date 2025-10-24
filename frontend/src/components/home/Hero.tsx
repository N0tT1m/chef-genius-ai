'use client'

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { SparklesIcon, ArrowRightIcon } from '@heroicons/react/24/outline'
import { motion } from 'framer-motion'

export function Hero() {
  const [ingredients, setIngredients] = useState('')
  const router = useRouter()

  const handleQuickGenerate = () => {
    if (ingredients.trim()) {
      const ingredientList = ingredients.split(',').map(s => s.trim()).filter(Boolean)
      router.push(`/recipes/generate?ingredients=${encodeURIComponent(ingredientList.join(','))}`)
    } else {
      router.push('/recipes/generate')
    }
  }

  return (
    <section className="relative overflow-hidden gradient-bg">
      <div className="mx-auto max-w-7xl px-4 py-24 sm:px-6 sm:py-32 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
          >
            <h1 className="text-4xl font-display font-bold tracking-tight text-gray-900 sm:text-6xl">
              Your AI-Powered
              <span className="text-primary-600"> Cooking Assistant</span>
            </h1>
          </motion.div>
          
          <motion.p 
            className="mt-6 text-lg leading-8 text-gray-600"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            Transform your culinary experience with intelligent recipe generation, 
            smart ingredient substitution, and personalized meal planning. 
            Create delicious meals tailored to your preferences and dietary needs.
          </motion.p>

          <motion.div 
            className="mt-10 flex flex-col items-center gap-6"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* Quick Recipe Generator */}
            <div className="w-full max-w-md">
              <div className="flex flex-col gap-3">
                <input
                  type="text"
                  placeholder="Enter ingredients (e.g., chicken, broccoli, rice)"
                  value={ingredients}
                  onChange={(e) => setIngredients(e.target.value)}
                  className="input-field"
                  onKeyDown={(e) => e.key === 'Enter' && handleQuickGenerate()}
                />
                <button
                  onClick={handleQuickGenerate}
                  className="btn-primary flex items-center justify-center gap-2 w-full"
                >
                  <SparklesIcon className="h-5 w-5" />
                  Generate Recipe
                  <ArrowRightIcon className="h-4 w-4" />
                </button>
              </div>
            </div>

            {/* Additional CTAs */}
            <div className="flex flex-col sm:flex-row gap-4">
              <button
                onClick={() => router.push('/meal-plans')}
                className="btn-outline"
              >
                Plan My Meals
              </button>
              <button
                onClick={() => router.push('/vision')}
                className="btn-secondary"
              >
                Scan Ingredients
              </button>
            </div>
          </motion.div>

          {/* Feature highlights */}
          <motion.div 
            className="mt-16 grid grid-cols-1 gap-6 sm:grid-cols-3"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.3 }}
          >
            <div className="text-center">
              <div className="text-2xl font-bold text-primary-600">10M+</div>
              <div className="text-sm text-gray-600">Recipes Generated</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary-600">500+</div>
              <div className="text-sm text-gray-600">Ingredients Recognized</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-primary-600">50+</div>
              <div className="text-sm text-gray-600">Cuisines Supported</div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Background decoration */}
      <div className="absolute inset-x-0 top-[calc(100%-13rem)] -z-10 transform-gpu overflow-hidden blur-3xl sm:top-[calc(100%-30rem)]">
        <div className="relative left-[calc(50%+3rem)] aspect-[1155/678] w-[36.125rem] -translate-x-1/2 bg-gradient-to-tr from-primary-300 to-secondary-300 opacity-20 sm:left-[calc(50%+36rem)] sm:w-[72.1875rem]" />
      </div>
    </section>
  )
}