# ChefGenius AI Mobile Demo

A Flutter mobile application demonstrating the capabilities of ChefGenius AI - your intelligent cooking assistant.

## Features

This demo app showcases four main AI-powered features:

### 🍳 Recipe Generator
- Input available ingredients
- Select cuisine preferences and dietary restrictions
- Generate custom recipes using AI
- Get detailed cooking instructions and ingredient lists

### 📷 Vision Analyzer
- Upload or take photos of food
- AI analyzes ingredients, cooking stages, and dish types
- Get confidence scores and detailed insights
- Receive recipe suggestions based on visual analysis

### 🧊 Fridge Scanner
- Scan your fridge contents with your camera
- AI identifies ingredients and assesses freshness
- Get organization tips and recipe suggestions
- Track items that need to be used soon

### 🤖 Chef Assistant
- Chat with an AI chef in real-time
- Send text messages or photos
- Get cooking advice, techniques, and tips
- Ask questions about ingredients and substitutions

## Technical Architecture

### Frontend (Flutter)
- **UI Framework**: Flutter with Material Design 3
- **State Management**: Provider pattern
- **Image Handling**: Image picker and camera integration
- **HTTP Client**: Custom API service with mock data support
- **Responsive Design**: Adaptive layouts for various screen sizes

### Backend Integration
The app integrates with ChefGenius AI backend APIs:
- `/recipes/generate` - Recipe generation
- `/vision/analyze-food-multimodal` - Image analysis
- `/vision/scan-fridge` - Fridge scanning
- `/chef/chat/*` - Conversational AI assistant

### Key Technologies
- **Flutter**: Cross-platform mobile development
- **Provider**: State management
- **JSON Serialization**: Automatic model generation
- **Image Processing**: Camera and gallery integration
- **Material Design 3**: Modern UI components

## Demo Mode

The app includes a comprehensive mock mode that demonstrates all features without requiring backend connectivity:

- **Mock Recipes**: Generated based on input ingredients
- **Mock Vision Analysis**: Simulated ingredient detection and analysis
- **Mock Fridge Scanning**: Sample fridge contents and suggestions
- **Mock Chat Responses**: Realistic AI chef conversations

## Project Structure

```
lib/
├── main.dart                 # App entry point
├── models/                   # Data models
│   └── recipe.dart          # Recipe and analysis models
├── services/                 # API and business logic
│   └── api_service.dart     # Backend integration
├── screens/                 # App screens
│   ├── home_screen.dart     # Main dashboard
│   ├── recipe_generator_screen.dart
│   ├── vision_analyzer_screen.dart
│   ├── fridge_scan_screen.dart
│   └── chef_chat_screen.dart
└── widgets/                 # Reusable UI components
    ├── recipe_card.dart
    └── loading_animation.dart
```

## Getting Started

### Prerequisites
- Flutter SDK (^3.5.4)
- Dart SDK
- iOS Simulator or Android Emulator (for testing)

### Installation
1. Clone the repository
2. Install dependencies:
   ```bash
   flutter pub get
   ```
3. Generate model files:
   ```bash
   dart run build_runner build
   ```
4. Run the app:
   ```bash
   flutter run
   ```

### Backend Configuration
To connect to a live backend:
1. Update `baseUrl` in `lib/services/api_service.dart`
2. Set `mockMode = false` in the same file
3. Ensure your backend is running and accessible

## Marketing Features

This demo app is specifically designed for marketing purposes:

### Professional Design
- Modern Material Design 3 interface
- Consistent branding with green color scheme
- Smooth animations and transitions
- Professional typography and spacing

### User Experience
- Intuitive navigation and feature discovery
- Clear feature demonstrations
- Engaging visual feedback
- Responsive design for all devices

### Feature Demonstrations
- Real-time AI capabilities showcase
- Interactive recipe generation
- Visual food analysis demonstrations
- Conversational AI interactions

### Call-to-Action Elements
- Feature highlight cards
- Demo mode indicators
- Engagement tracking ready
- Social sharing capabilities (placeholder)

## Platform Support

- ✅ iOS
- ✅ Android
- ✅ macOS (for demo purposes)
- 🔄 Web (can be enabled)

## Development Notes

### Performance
- Optimized image handling for mobile devices
- Efficient state management with Provider
- Lazy loading for better performance
- Cached network images

### Security
- No sensitive data stored locally
- Secure API communication (when connected)
- Proper error handling and fallbacks

### Accessibility
- Semantic labels for screen readers
- High contrast color schemes
- Keyboard navigation support
- Scalable text and UI elements

## Future Enhancements

- [ ] Offline recipe storage
- [ ] User accounts and preferences
- [ ] Recipe sharing functionality
- [ ] Advanced camera features
- [ ] Voice interaction
- [ ] Nutrition tracking
- [ ] Shopping list generation
- [ ] Meal planning features

## License

This is a demo application for marketing purposes. All rights reserved.

## Contact

For more information about ChefGenius AI, visit our website or contact our sales team.