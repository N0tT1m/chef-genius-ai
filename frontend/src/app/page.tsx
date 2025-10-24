import { Hero } from '@/components/home/Hero'
import { Features } from '@/components/home/Features'
import { HowItWorks } from '@/components/home/HowItWorks'
import { Testimonials } from '@/components/home/Testimonials'
import { CTA } from '@/components/home/CTA'

export default function HomePage() {
  return (
    <div className="space-y-16">
      <Hero />
      <Features />
      <HowItWorks />
      <Testimonials />
      <CTA />
    </div>
  )
}