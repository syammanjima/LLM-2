import streamlit as st
import PIL.Image as Image
from PIL import ImageEnhance, ImageFilter
import io
import base64
from datetime import datetime
import numpy as np
import pandas as pd
import cv2
import json
import re

# Configure the page
st.set_page_config(
    page_title="Multi-Modal Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I'm your Multi-Modal Assistant. I can help you with text-based questions AND analyze images. Upload an image and ask me anything about it, or just chat with me about any topic!",
            "timestamp": datetime.now(),
            "has_image": False
        }
    ]

if 'current_image' not in st.session_state:
    st.session_state.current_image = None

if 'image_analysis' not in st.session_state:
    st.session_state.image_analysis = None

# Knowledge base for text-based queries
KNOWLEDGE_BASE = {
    'ai_ml': {
        'machine learning': 'Machine Learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.',
        'artificial intelligence': 'AI is the simulation of human intelligence in machines that are programmed to think and learn like humans.',
        'neural networks': 'Neural networks are computing systems inspired by biological neural networks, used in machine learning.',
        'deep learning': 'Deep learning is a subset of machine learning with neural networks that have three or more layers.'
    },
    'programming': {
        'python': 'Python is a high-level, interpreted programming language known for its simplicity and versatility.',
        'javascript': 'JavaScript is a programming language primarily used for web development and creating interactive web pages.',
        'java': 'Java is a popular object-oriented programming language known for its "write once, run anywhere" philosophy.',
        'streamlit': 'Streamlit is a Python framework for building interactive web applications for data science and ML projects.',
        'opencv': 'OpenCV is a library of programming functions mainly aimed at real-time computer vision.'
    },
    'java_howto': {
        'how to run java': '''**How to Run a Java Program:**

1. **Write your Java code** in a .java file (e.g., HelloWorld.java)
2. **Compile:** `javac HelloWorld.java` (creates HelloWorld.class)
3. **Run:** `java HelloWorld` (executes the bytecode)

**Prerequisites:** Install JDK (Java Development Kit) from Oracle or OpenJDK

**Example:**
```java
public class HelloWorld {
    public static void main(String[] args) {
        System.out.println("Hello, World!");
    }
}
```

**Commands:**
- Compile: `javac HelloWorld.java`
- Run: `java HelloWorld`''',
        'how to run a java program': '''**Step-by-Step Guide to Run Java Programs:**

**Method 1: Command Line**
1. Install JDK (Java Development Kit)
2. Write code in .java file
3. Open terminal/command prompt
4. Navigate to file location: `cd /path/to/your/file`
5. Compile: `javac YourProgram.java`
6. Run: `java YourProgram`

**Method 2: IDE (Recommended)**
- Use IntelliJ IDEA, Eclipse, or VS Code
- Create new Java project
- Write code and click "Run" button

**Common Issues:**
- Ensure JAVA_HOME is set
- Check PATH includes Java bin directory
- File name must match class name exactly''',
        'java compile': 'To compile Java: Use `javac FileName.java` command. This creates a .class bytecode file that can be executed with `java FileName`.',
        'java execution': 'To execute Java: Use `java ClassName` (without .class extension) after compiling with javac.'
    },
    'general': {
        'what is': 'I can help explain concepts, analyze images, answer questions, and assist with various topics.',
        'how to': 'I can provide step-by-step guidance on many topics. What would you like to learn how to do?',
        'explain': 'I can explain complex topics in simple terms. What would you like me to explain?'
    }
}

def analyze_image_content(image):
    """Comprehensive image analysis"""
    try:
        # Convert PIL to OpenCV
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Basic image properties
        height, width = img_array.shape[:2]
        total_pixels = height * width
        
        # Color analysis
        hsv = cv2.cvtColor(img_array, cv2.COLOR_BGR2HSV)
        mean_hsv = np.mean(hsv, axis=(0, 1))
        
        # Brightness analysis
        brightness = np.mean(img_array)
        brightness_desc = get_brightness_description(brightness)
        
        # Color analysis
        dominant_colors = analyze_colors(img_array)
        
        # Object detection
        objects = detect_objects(img_array, hsv)
        
        # Scene analysis
        scene_type = determine_scene_type(objects, brightness, dominant_colors)
        
        # Composition analysis
        composition = analyze_composition(img_array)
        
        return {
            'dimensions': f"{width}×{height}",
            'total_pixels': total_pixels,
            'brightness': brightness,
            'brightness_desc': brightness_desc,
            'dominant_colors': dominant_colors,
            'objects': objects,
            'scene_type': scene_type,
            'composition': composition,
            'technical_quality': assess_technical_quality(img_array)
        }
        
    except Exception as e:
        return {'error': f"Analysis error: {str(e)}"}

def get_brightness_description(brightness):
    """Describe brightness level"""
    if brightness > 200: return "Very bright"
    elif brightness > 150: return "Bright"
    elif brightness > 100: return "Well-lit"
    elif brightness > 50: return "Moderately lit"
    else: return "Dark"

def analyze_colors(img_array):
    """Analyze dominant colors in the image"""
    # Reshape for color analysis
    pixels = img_array.reshape(-1, 3)
    
    # Simple color categorization
    mean_color = np.mean(pixels, axis=0)
    b, g, r = mean_color
    
    color_desc = []
    
    if g > r * 1.3 and g > b * 1.3:
        color_desc.append("Green/Nature tones")
    if b > r * 1.3 and b > g * 1.3:
        color_desc.append("Blue/Cool tones")
    if r > g * 1.3 and r > b * 1.3:
        color_desc.append("Red/Warm tones")
    if r > 180 and g > 180 and b > 180:
        color_desc.append("Light/Bright colors")
    if r < 80 and g < 80 and b < 80:
        color_desc.append("Dark colors")
    
    if not color_desc:
        color_desc.append("Mixed colors")
    
    return color_desc

def detect_objects(img_array, hsv):
    """Detect various objects in the image"""
    objects = []
    
    # Trees/Vegetation detection
    green_mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
    green_percentage = (np.sum(green_mask > 0) / green_mask.size) * 100
    
    if green_percentage > 20:
        objects.append("Dense vegetation/trees")
    elif green_percentage > 10:
        objects.append("Vegetation/plants")
    elif green_percentage > 5:
        objects.append("Some greenery")
    
    # Sky detection
    blue_mask = cv2.inRange(hsv, np.array([100, 50, 100]), np.array([130, 255, 255]))
    blue_percentage = (np.sum(blue_mask > 0) / blue_mask.size) * 100
    
    if blue_percentage > 15:
        objects.append("Sky/open air")
    
    # Water detection
    water_mask = cv2.inRange(hsv, np.array([90, 50, 50]), np.array([120, 255, 200]))
    water_percentage = (np.sum(water_mask > 0) / water_mask.size) * 100
    
    if water_percentage > 10:
        objects.append("Water body")
    
    # Building/Structure detection
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None and len(lines) > 20:
        objects.append("Buildings/structures")
    elif lines is not None and len(lines) > 10:
        objects.append("Architectural elements")
    
    # People detection (simplified)
    skin_mask = cv2.inRange(hsv, np.array([0, 20, 70]), np.array([20, 255, 255]))
    skin_percentage = (np.sum(skin_mask > 0) / skin_mask.size) * 100
    
    if skin_percentage > 8:
        objects.append("People")
    elif skin_percentage > 3:
        objects.append("Possible people")
    
    return objects if objects else ["General scene"]

def determine_scene_type(objects, brightness, colors):
    """Determine the type of scene"""
    if any("vegetation" in obj or "trees" in obj for obj in objects):
        if "Sky" in str(objects):
            return "Outdoor nature scene"
        else:
            return "Nature/forest scene"
    elif "Sky" in str(objects):
        return "Outdoor scene"
    elif any("building" in obj.lower() or "structure" in obj.lower() for obj in objects):
        return "Urban/architectural scene"
    elif "People" in str(objects):
        return "Portrait/people scene"
    elif brightness < 100:
        return "Indoor scene"
    else:
        return "General scene"

def analyze_composition(img_array):
    """Analyze image composition"""
    height, width = img_array.shape[:2]
    
    # Rule of thirds analysis
    thirds_h = height // 3
    thirds_w = width // 3
    
    # Analyze brightness distribution
    top_third = np.mean(img_array[:thirds_h, :])
    middle_third = np.mean(img_array[thirds_h:2*thirds_h, :])
    bottom_third = np.mean(img_array[2*thirds_h:, :])
    
    composition_notes = []
    
    if top_third > middle_third * 1.2:
        composition_notes.append("Bright upper region (likely sky)")
    if bottom_third < middle_third * 0.8:
        composition_notes.append("Darker lower region (foreground/ground)")
    
    # Edge analysis for focus
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    
    if edge_density > 0.1:
        composition_notes.append("High detail/sharp focus")
    elif edge_density < 0.03:
        composition_notes.append("Soft focus/low detail")
    else:
        composition_notes.append("Moderate detail level")
    
    return composition_notes

def assess_technical_quality(img_array):
    """Assess technical quality of the image"""
    # Brightness distribution
    brightness = np.mean(img_array)
    brightness_std = np.std(img_array)
    
    quality_notes = []
    
    if brightness_std > 60:
        quality_notes.append("Good contrast")
    elif brightness_std < 30:
        quality_notes.append("Low contrast")
    
    if 50 < brightness < 200:
        quality_notes.append("Well-exposed")
    elif brightness < 50:
        quality_notes.append("Underexposed")
    elif brightness > 200:
        quality_notes.append("Bright/possibly overexposed")
    
    return quality_notes

def process_text_query(query):
    """Process text-based queries using knowledge base"""
    query_lower = query.lower()
    
    # Check for specific Java how-to questions first
    if 'how to run java' in query_lower or 'run a java program' in query_lower:
        if 'program' in query_lower:
            return KNOWLEDGE_BASE['java_howto']['how to run a java program']
        else:
            return KNOWLEDGE_BASE['java_howto']['how to run java']
    
    if 'java compile' in query_lower or 'compile java' in query_lower:
        return KNOWLEDGE_BASE['java_howto']['java compile']
    
    if 'java execution' in query_lower or 'execute java' in query_lower:
        return KNOWLEDGE_BASE['java_howto']['java execution']
    
    # Check knowledge base
    for category, items in KNOWLEDGE_BASE.items():
        for key, value in items.items():
            if key in query_lower:
                if category == 'java_howto':
                    return value
                else:
                    return f"**{key.title()}**: {value}"
    
    # Handle specific question types
    if any(word in query_lower for word in ['what is', 'what are', 'define']):
        topic = extract_topic_from_query(query_lower)
        if 'java' in topic:
            return f"**Java Programming**: {KNOWLEDGE_BASE['programming']['java']}\n\nWould you like to know how to run Java programs? Just ask 'how to run java'!"
        return f"I'd be happy to explain **{topic}**! Could you be more specific about what aspect you'd like to know about?"
    
    elif any(word in query_lower for word in ['how to', 'how do i', 'how can i']):
        if 'java' in query_lower:
            return KNOWLEDGE_BASE['java_howto']['how to run java']
        return "I can help you with step-by-step instructions! I'm especially good with Java, Python, AI/ML topics. What specifically would you like to learn?"
    
    elif any(word in query_lower for word in ['explain', 'tell me about']):
        topic = extract_topic_from_query(query_lower)
        if 'java' in topic:
            return f"**Java Programming**: {KNOWLEDGE_BASE['programming']['java']}\n\n**Want to learn how to run Java programs?** Ask me 'how to run java' for step-by-step instructions!"
        return f"I can explain **{topic}** for you! What specific aspect would you like me to focus on?"
    
    elif any(word in query_lower for word in ['hello', 'hi', 'hey']):
        return "Hello! I'm here to help with both text questions and image analysis. I'm especially good with programming (Java, Python), AI/ML topics, and image analysis. What can I assist you with today?"
    
    elif any(word in query_lower for word in ['help', 'what can you do']):
        return """**I can help you with:**

**💻 Programming:**
• **Java**: How to run programs, compile, execute
• **Python**: Syntax, libraries, best practices  
• **JavaScript**: Web development, frameworks
• **General**: Coding concepts and guidance

**🤖 AI & Machine Learning:**
• Machine Learning concepts and applications
• Neural Networks and Deep Learning
• AI fundamentals and terminology

**🖼️ Image Analysis:**
• Upload images for automatic analysis
• Object detection (trees, buildings, people, etc.)
• Color, lighting, and composition analysis
• Scene type classification

**🔗 Multi-Modal:**
• Combine programming questions with image analysis
• Answer questions about code screenshots
• Analyze technical diagrams or flowcharts

**Try asking:**
• "How to run a Java program?"
• "What is machine learning?"
• "Explain Python" 
• Or upload an image!"""
    
    else:
        # Check if it's a simple topic mention
        if 'java' in query_lower:
            return f"**Java Programming**: {KNOWLEDGE_BASE['programming']['java']}\n\n**Need help running Java programs?** Ask me 'how to run java' for detailed instructions!"
        
        return f"I understand you're asking about: \"{query}\"\n\nI can help with **programming** (Java, Python, JavaScript), **AI/ML topics**, and **image analysis**. Could you be more specific, or upload an image if you'd like visual analysis?"

def extract_topic_from_query(query):
    """Extract main topic from query"""
    # Remove common question words
    topic = re.sub(r'\b(what is|what are|tell me about|explain|how to|how do i|how can i)\b', '', query)
    topic = topic.strip().strip('?').strip()
    return topic if topic else "that topic"

def generate_image_description(analysis):
    """Generate natural language description of image"""
    if 'error' in analysis:
        return f"❌ Could not analyze image: {analysis['error']}"
    
    description = f"""🖼️ **Image Analysis:**

**📏 Technical Details:**
• Dimensions: {analysis['dimensions']} pixels
• Lighting: {analysis['brightness_desc']} (brightness: {analysis['brightness']:.1f}/255)
• Colors: {', '.join(analysis['dominant_colors'])}

**🎯 Content Detected:**
• Objects/Elements: {', '.join(analysis['objects'])}
• Scene Type: {analysis['scene_type']}

**🎨 Composition:**
• {', '.join(analysis['composition'])}

**📊 Technical Quality:**
• {', '.join(analysis['technical_quality'])}
"""
    return description

def answer_image_question(query, analysis):
    """Answer specific questions about the image"""
    if 'error' in analysis:
        return f"I couldn't analyze the image properly. Error: {analysis['error']}"
    
    query_lower = query.lower()
    
    # Specific object questions
    if any(word in query_lower for word in ['tree', 'trees', 'vegetation', 'plants']):
        vegetation_objects = [obj for obj in analysis['objects'] if 'vegetation' in obj.lower() or 'tree' in obj.lower() or 'green' in obj.lower()]
        if vegetation_objects:
            return f"🌳 **Yes, I can see vegetation!** Specifically: {', '.join(vegetation_objects)}"
        else:
            return f"🌳 **No clear vegetation detected** in this image. The scene appears to be: {analysis['scene_type']}"
    
    elif any(word in query_lower for word in ['people', 'person', 'human', 'face']):
        people_objects = [obj for obj in analysis['objects'] if 'people' in obj.lower() or 'person' in obj.lower()]
        if people_objects:
            return f"👥 **Yes, I can detect people!** Specifically: {', '.join(people_objects)}"
        else:
            return f"👥 **No clear people detected** in this image."
    
    elif any(word in query_lower for word in ['building', 'house', 'structure', 'architecture']):
        building_objects = [obj for obj in analysis['objects'] if 'building' in obj.lower() or 'structure' in obj.lower() or 'architectural' in obj.lower()]
        if building_objects:
            return f"🏢 **Yes, I can see structures!** Specifically: {', '.join(building_objects)}"
        else:
            return f"🏢 **No clear buildings detected** in this image."
    
    elif any(word in query_lower for word in ['color', 'colors']):
        return f"🎨 **Colors in this image:** {', '.join(analysis['dominant_colors'])}\n\n**Lighting:** {analysis['brightness_desc']}\n**Overall tone:** {analysis['scene_type']}"
    
    elif any(word in query_lower for word in ['bright', 'dark', 'lighting', 'light']):
        return f"💡 **Lighting Analysis:**\n• **Brightness Level:** {analysis['brightness_desc']} ({analysis['brightness']:.1f}/255)\n• **Technical Quality:** {', '.join(analysis['technical_quality'])}"
    
    elif any(word in query_lower for word in ['what do you see', 'describe', 'what is in', 'analyze']):
        return generate_image_description(analysis)
    
    elif any(word in query_lower for word in ['scene', 'type', 'setting']):
        return f"🏞️ **Scene Analysis:**\n• **Type:** {analysis['scene_type']}\n• **Elements:** {', '.join(analysis['objects'])}\n• **Composition:** {', '.join(analysis['composition'])}"
    
    else:
        # General response combining image context with query
        return f"**Your question:** \"{query}\"\n\n**About this image:**\n{generate_image_description(analysis)}\n\n**Could you ask something more specific about what you see in the image?**"

def generate_response(query, has_image=False, image_analysis=None):
    """Main response generation function"""
    
    # If no query but there's an image, provide automatic analysis
    if not query.strip() and has_image and image_analysis:
        return f"I've analyzed your image! Here's what I found:\n\n{generate_image_description(image_analysis)}\n\n**Ask me anything about this image!**"
    
    # If no query and no image
    if not query.strip():
        return "Hello! I'm your Multi-Modal Assistant. I can help with text questions or analyze images. What would you like to know?"
    
    # If there's an image and a query, prioritize image-related response
    if has_image and image_analysis:
        # Check if query is about the image
        if any(word in query.lower() for word in ['image', 'picture', 'photo', 'this', 'see', 'show', 'visible', 'in it']):
            return answer_image_question(query, image_analysis)
        else:
            # Provide both text response and image context
            text_response = process_text_query(query)
            return f"{text_response}\n\n**Note:** I also have an image loaded if you'd like to ask about it!"
    
    # Pure text query
    return process_text_query(query)

def analyze_uploaded_image(image_file):
    """Analyze uploaded image file"""
    try:
        image = Image.open(image_file)
        
        # Metadata
        metadata = {
            "filename": image_file.name,
            "format": image.format,
            "mode": image.mode,
            "size": image.size,
            "width": image.size[0],
            "height": image.size[1],
            "file_size": len(image_file.getvalue()),
        }
        
        # Content analysis
        content_analysis = analyze_image_content(image)
        
        return metadata, image, content_analysis
        
    except Exception as e:
        st.error(f"Error analyzing image: {str(e)}")
        return None, None, None

# Sidebar for image upload and options
with st.sidebar:
    st.header("🖼️ Image Upload")
    
    uploaded_file = st.file_uploader(
        "Upload an image for analysis",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
        help="Upload any image and ask questions about it"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔍 Analyzing image..."):
            metadata, image, analysis = analyze_uploaded_image(uploaded_file)
            
        if metadata and image and analysis:
            st.session_state.current_image = metadata
            st.session_state.image_analysis = analysis
            
            # Display image
            st.image(image, caption=f"📸 {metadata['filename']}", use_container_width=True)
            
            # Show quick stats
            st.success(f"""**Image Loaded:**
            • Size: {metadata['width']}×{metadata['height']}
            • Format: {metadata['format']}
            • File: {metadata['file_size']/1024:.1f} KB""")
            
            # Auto-generate analysis message
            auto_analysis = generate_image_description(analysis)
            st.session_state.messages.append({
                "role": "assistant",
                "content": f"I've analyzed your image '{metadata['filename']}':\n\n{auto_analysis}",
                "timestamp": datetime.now(),
                "has_image": True
            })
            
            if st.button("🗑️ Remove Image"):
                st.session_state.current_image = None
                st.session_state.image_analysis = None
                st.rerun()
    
    elif st.session_state.current_image:
        st.info(f"📸 Current image: {st.session_state.current_image['filename']}")
        if st.button("🗑️ Remove Image"):
            st.session_state.current_image = None
            st.session_state.image_analysis = None
            st.rerun()
    
    st.divider()
    
    # Quick action buttons
    st.header("⚡ Quick Actions")
    
    if st.session_state.current_image:
        image_actions = [
            "What do you see in this image?",
            "Describe the colors and lighting",
            "What objects can you identify?",
            "Analyze the scene type"
        ]
        
        for action in image_actions:
            if st.button(action, key=f"img_{action}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": action, 
                    "timestamp": datetime.now(),
                    "has_image": True
                })
                st.rerun()
    else:
        text_actions = [
            "How to run a Java program?",
            "What is machine learning?",
            "Explain Python programming",
            "What can you help me with?"
        ]
        
        for action in text_actions:
            if st.button(action, key=f"txt_{action}", use_container_width=True):
                st.session_state.messages.append({
                    "role": "user", 
                    "content": action, 
                    "timestamp": datetime.now(),
                    "has_image": False
                })
                st.rerun()

# Main interface
st.title("🤖 Multi-Modal Assistant")
st.markdown("*I can help with text questions AND analyze images!*")

# Show current mode
if st.session_state.current_image:
    st.info(f"🖼️ **Image Mode**: Currently analyzing '{st.session_state.current_image['filename']}' - Ask me about it!")
else:
    st.info("💬 **Text Mode**: Ask me questions or upload an image for multi-modal analysis!")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        st.caption(f"*{message['timestamp'].strftime('%H:%M:%S')} - {'Image Mode' if message.get('has_image') else 'Text Mode'}*")

# Chat input
if prompt := st.chat_input("Ask me anything or upload an image..."):
    # Determine if this is image-related
    has_image_context = st.session_state.current_image is not None
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": datetime.now(),
        "has_image": has_image_context
    })
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            response = generate_response(
                prompt, 
                has_image_context, 
                st.session_state.image_analysis
            )
            st.markdown(response)
            
            # Add to message history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "timestamp": datetime.now(),
                "has_image": has_image_context
            })

# Footer with controls
st.divider()

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("💬 Messages", len(st.session_state.messages))

with col2:
    mode = "Image Mode" if st.session_state.current_image else "Text Mode"
    st.metric("🔄 Current Mode", mode)

with col3:
    if st.button("🗑️ Clear Chat"):
        # Keep only the welcome message
        st.session_state.messages = [st.session_state.messages[0]]
        st.rerun()

# Help section
with st.expander("ℹ️ How to Use This Multi-Modal Assistant"):
    st.markdown("""
    **🤖 Multi-Modal Capabilities:**
    
    **💬 Text Mode:**
    • Ask questions about AI, machine learning, programming
    • Get explanations of concepts and definitions
    • Receive step-by-step guidance
    • General conversation and assistance
    
    **🖼️ Image Mode:**
    • Upload any image (PNG, JPG, etc.)
    • Get automatic analysis of objects, colors, lighting
    • Ask specific questions about what's in the image
    • Analyze scene types, composition, and technical quality
    
    **🔗 Multi-Modal Features:**
    • Combine text questions with image analysis
    • Ask follow-up questions about uploaded images
    • Switch between text and image modes seamlessly
    • Get contextual responses based on current mode
    
    **💡 Example Interactions:**
    • "What is machine learning?" (Text)
    • Upload photo → "What do you see in this image?" (Image)
    • "Are there any trees in this photo?" (Multi-modal)
    • "Explain the colors in this image" (Multi-modal)
    """)
