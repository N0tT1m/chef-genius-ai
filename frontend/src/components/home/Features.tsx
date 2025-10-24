'use client'

import { motion } from 'framer-motion'
import {
  SparklesIcon,
  ArrowPathIcon,
  CalendarDaysIcon,
  CameraIcon,
  ChartBarIcon,
  ClockIcon,
  HeartIcon,
  GlobeAltIcon,
} from '@heroicons/react/24/outline'

const features = [
  {
    name: 'AI Recipe Generation',
    description: 'Create unique recipes based on your available ingredients, dietary preferences, and cooking skills.',
    icon: SparklesIcon,
    color: 'primary',
  },
  {
    name: 'Smart Substitutions',
    description: 'Find perfect ingredient replacements that maintain flavor profiles and dietary requirements.',
    icon: ArrowPathIcon,
    color: 'secondary',
  },
  {
    name: 'Meal Planning',
    description: 'Generate balanced weekly meal plans optimized for nutrition, budget, and preparation time.',
    icon: CalendarDaysIcon,
    color: 'accent',
  },
  {
    name: 'Visual Recognition',
    description: 'Scan ingredients with your camera and get instant recipe suggestions and cooking guidance.',
    icon: CameraIcon,
    color: 'primary',
  },
  {
    name: 'Nutrition Analysis',
    description: 'Get detailed nutritional information and optimize recipes for your health goals.',
    icon: ChartBarIcon,
    color: 'secondary',
  },
  {
    name: 'Time Optimization',
    description: 'Smart cooking schedules and prep suggestions to save time in the kitchen.',
    icon: ClockIcon,
    color: 'accent',
  },
  {
    name: 'Dietary Adaptation',
    description: 'Automatically adapt any recipe to fit vegan, keto, gluten-free, and other dietary needs.',
    icon: HeartIcon,
    color: 'primary',
  },
  {
    name: 'Global Cuisines',
    description: 'Explore and create authentic dishes from 50+ international cuisines and cooking styles.',
    icon: GlobeAltIcon,
    color: 'secondary',
  },
]

export function Features() {
  return (
    <section className="py-24 sm:py-32">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <motion.h2 
            className="text-3xl font-display font-bold tracking-tight text-gray-900 sm:text-4xl"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            Everything you need to become a better cook
          </motion.h2>
          <motion.p 
            className="mt-6 text-lg leading-8 text-gray-600"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            ChefGenius combines cutting-edge AI with culinary expertise to provide 
            a comprehensive cooking assistant that adapts to your needs and preferences.
          </motion.p>
        </div>

        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <dl className="grid max-w-xl grid-cols-1 gap-x-8 gap-y-16 lg:max-w-none lg:grid-cols-4">
            {features.map((feature, index) => (
              <motion.div 
                key={feature.name}
                className="flex flex-col"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
              >
                <dt className="flex items-center gap-x-3 text-base font-semibold leading-7 text-gray-900">
                  <div className={`h-10 w-10 flex items-center justify-center rounded-lg bg-${feature.color}-600`}>
                    <feature.icon className="h-6 w-6 text-white" aria-hidden="true" />
                  </div>
                  {feature.name}
                </dt>
                <dd className="mt-4 flex flex-auto flex-col text-base leading-7 text-gray-600">
                  <p className="flex-auto">{feature.description}</p>
                </dd>
              </motion.div>
            ))}
          </dl>
        </div>
      </div>
    </section>
  )
}