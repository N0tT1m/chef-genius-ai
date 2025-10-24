'use client'

import { motion } from 'framer-motion'
import {
  DocumentTextIcon,
  SparklesIcon,
  ClipboardDocumentCheckIcon,
} from '@heroicons/react/24/outline'

const steps = [
  {
    name: 'Share Your Preferences',
    description: 'Tell us what ingredients you have, your dietary needs, cuisine preferences, and cooking time.',
    icon: DocumentTextIcon,
    step: '01',
  },
  {
    name: 'AI Creates Your Recipe',
    description: 'Our advanced AI analyzes your inputs and generates personalized recipes with detailed instructions.',
    icon: SparklesIcon,
    step: '02',
  },
  {
    name: 'Cook & Enjoy',
    description: 'Follow the step-by-step instructions, get real-time cooking tips, and enjoy your delicious meal.',
    icon: ClipboardDocumentCheckIcon,
    step: '03',
  },
]

export function HowItWorks() {
  return (
    <section className="relative py-24 sm:py-32 bg-gray-50">
      <div className="mx-auto max-w-7xl px-6 lg:px-8">
        <div className="mx-auto max-w-2xl text-center">
          <motion.h2 
            className="text-3xl font-display font-bold tracking-tight text-gray-900 sm:text-4xl"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
          >
            How ChefGenius Works
          </motion.h2>
          <motion.p 
            className="mt-6 text-lg leading-8 text-gray-600"
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            viewport={{ once: true }}
          >
            Getting started with ChefGenius is simple. Just follow these three easy steps 
            to transform your cooking experience.
          </motion.p>
        </div>

        <div className="mx-auto mt-16 max-w-2xl sm:mt-20 lg:mt-24 lg:max-w-none">
          <div className="grid max-w-xl grid-cols-1 gap-y-16 lg:max-w-none lg:grid-cols-3 lg:gap-x-8">
            {steps.map((step, index) => (
              <motion.div 
                key={step.name}
                className="flex flex-col items-center text-center"
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.2 }}
                viewport={{ once: true }}
              >
                <div className="relative">
                  <div className="flex h-20 w-20 items-center justify-center rounded-full bg-primary-600">
                    <step.icon className="h-10 w-10 text-white" aria-hidden="true" />
                  </div>
                  <div className="absolute -top-2 -right-2 flex h-8 w-8 items-center justify-center rounded-full bg-primary-100 text-sm font-bold text-primary-600">
                    {step.step}
                  </div>
                </div>
                
                <h3 className="mt-6 text-xl font-display font-semibold text-gray-900">
                  {step.name}
                </h3>
                <p className="mt-4 text-base text-gray-600 max-w-sm">
                  {step.description}
                </p>

                {/* Connector line */}
                {index < steps.length - 1 && (
                  <div className="hidden lg:block absolute top-10 left-1/2 w-full h-0.5 bg-gradient-to-r from-primary-300 to-primary-200 transform translate-x-10" />
                )}
              </motion.div>
            ))}
          </div>
        </div>

        {/* Call to action */}
        <motion.div 
          className="mt-16 text-center"
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
          viewport={{ once: true }}
        >
          <button className="btn-primary text-lg px-8 py-3">
            Start Cooking Smarter Today
          </button>
        </motion.div>
      </div>
    </section>
  )
}