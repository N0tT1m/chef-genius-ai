'use client'

import { motion } from 'framer-motion'
import { useRouter } from 'next/navigation'
import { SparklesIcon, ArrowRightIcon } from '@heroicons/react/24/outline'

export function CTA() {
  const router = useRouter()

  return (
    <section className="relative isolate overflow-hidden bg-primary-600 py-16 sm:py-24 lg:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto grid max-w-2xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-2">
          <motion.div 
            className="max-w-xl lg:max-w-lg"
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            <h2 className="text-3xl font-display font-bold tracking-tight text-white sm:text-4xl">
              Ready to revolutionize your cooking?
            </h2>
            <p className="mt-4 text-lg leading-8 text-primary-100">
              Join thousands of cooks who have transformed their kitchen experience with ChefGenius. 
              Start creating personalized recipes, planning balanced meals, and cooking with confidence today.
            </p>
            <div className="mt-6 flex flex-col gap-4 sm:flex-row">
              <button
                onClick={() => router.push('/recipes/generate')}
                className="flex items-center justify-center gap-2 rounded-md bg-white px-6 py-3 text-sm font-semibold text-primary-600 shadow-sm hover:bg-primary-50 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-primary-600 transition-colors"
              >
                <SparklesIcon className="h-5 w-5" />
                Start Generating Recipes
                <ArrowRightIcon className="h-4 w-4" />
              </button>
              <button
                onClick={() => router.push('/about')}
                className="rounded-md border border-primary-300 px-6 py-3 text-sm font-semibold text-white hover:bg-primary-500 focus:outline-none focus:ring-2 focus:ring-white focus:ring-offset-2 focus:ring-offset-primary-600 transition-colors"
              >
                Learn More
              </button>
            </div>
          </motion.div>
          
          <motion.div 
            className="lg:flex lg:justify-end"
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            <div className="w-full max-w-md">
              <div className="card bg-white/10 backdrop-blur-sm border-primary-300">
                <h3 className="text-lg font-semibold text-white mb-4">
                  Quick Recipe Preview
                </h3>
                <div className="space-y-3">
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-2 rounded-full bg-primary-300"></div>
                    <span className="text-primary-100 text-sm">Ingredient-based generation</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-2 rounded-full bg-primary-300"></div>
                    <span className="text-primary-100 text-sm">Dietary customization</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-2 rounded-full bg-primary-300"></div>
                    <span className="text-primary-100 text-sm">Nutrition analysis</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <div className="h-2 w-2 rounded-full bg-primary-300"></div>
                    <span className="text-primary-100 text-sm">Step-by-step instructions</span>
                  </div>
                </div>
                <div className="mt-6 p-3 bg-white/5 rounded-lg">
                  <div className="text-xs text-primary-200 uppercase tracking-wide font-medium">
                    Example Recipe
                  </div>
                  <div className="text-sm text-white mt-1">
                    "Mediterranean Herb-Crusted Salmon with Roasted Vegetables"
                  </div>
                  <div className="text-xs text-primary-200 mt-1">
                    Generated in 2.3 seconds
                  </div>
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
      
      {/* Background decoration */}
      <div className="absolute left-1/2 top-0 -z-10 -translate-x-1/2 blur-3xl xl:-top-6" aria-hidden="true">
        <div className="aspect-[1155/678] w-[72.1875rem] bg-gradient-to-tr from-primary-300 to-secondary-300 opacity-30" />
      </div>
    </section>
  )
}