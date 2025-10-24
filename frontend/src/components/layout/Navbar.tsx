'use client'

import { useState } from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { 
  ChefHatIcon, 
  Bars3Icon, 
  XMarkIcon,
  SparklesIcon,
  BookOpenIcon,
  CalendarDaysIcon,
  CameraIcon
} from '@heroicons/react/24/outline'

export function Navbar() {
  const [isOpen, setIsOpen] = useState(false)
  const router = useRouter()

  const navigation = [
    { name: 'Generate Recipe', href: '/recipes/generate', icon: SparklesIcon },
    { name: 'Browse Recipes', href: '/recipes', icon: BookOpenIcon },
    { name: 'Meal Plans', href: '/meal-plans', icon: CalendarDaysIcon },
    { name: 'Visual Assistant', href: '/vision', icon: CameraIcon },
  ]

  return (
    <nav className="bg-white shadow-sm border-b border-gray-200">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8">
        <div className="flex h-16 justify-between items-center">
          {/* Logo */}
          <Link href="/" className="flex items-center space-x-2">
            <ChefHatIcon className="h-8 w-8 text-primary-600" />
            <span className="text-xl font-display font-bold text-gray-900">
              ChefGenius
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            {navigation.map((item) => (
              <Link
                key={item.name}
                href={item.href}
                className="flex items-center space-x-2 text-sm font-medium text-gray-700 hover:text-primary-600 transition-colors"
              >
                <item.icon className="h-4 w-4" />
                <span>{item.name}</span>
              </Link>
            ))}
          </div>

          {/* CTA Button */}
          <div className="hidden md:flex items-center space-x-4">
            <button
              onClick={() => router.push('/recipes/generate')}
              className="btn-primary"
            >
              Get Started
            </button>
          </div>

          {/* Mobile menu button */}
          <div className="md:hidden">
            <button
              type="button"
              className="inline-flex items-center justify-center rounded-md p-2 text-gray-700 hover:bg-gray-100 hover:text-primary-600 focus:outline-none focus:ring-2 focus:ring-inset focus:ring-primary-500"
              onClick={() => setIsOpen(!isOpen)}
            >
              <span className="sr-only">Open main menu</span>
              {isOpen ? (
                <XMarkIcon className="block h-6 w-6" />
              ) : (
                <Bars3Icon className="block h-6 w-6" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isOpen && (
          <div className="md:hidden border-t border-gray-200 pt-4 pb-3">
            <div className="space-y-1">
              {navigation.map((item) => (
                <Link
                  key={item.name}
                  href={item.href}
                  className="flex items-center space-x-3 px-3 py-2 text-base font-medium text-gray-700 hover:bg-gray-50 hover:text-primary-600 rounded-md"
                  onClick={() => setIsOpen(false)}
                >
                  <item.icon className="h-5 w-5" />
                  <span>{item.name}</span>
                </Link>
              ))}
              <div className="px-3 py-2">
                <button
                  onClick={() => {
                    setIsOpen(false)
                    router.push('/recipes/generate')
                  }}
                  className="btn-primary w-full"
                >
                  Get Started
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </nav>
  )
}