import streamlit as st
import speech_recognition as sr
import tempfile
import os
import json
from datetime import datetime
import time
import re

st.set_page_config(page_title="Meeting Notes Extractor", page_icon="🎤", layout="wide")

st.title("🎤 Meeting Notes Extractor")
st.markdown("**Works with WAV files - No FFmpeg required!**")

def transcribe_wav_file(wav_file):
    """Transcribe WAV file with better error handling and debugging"""
    try:
        recognizer = sr.Recognizer()
        
        # Save uploaded file to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(wav_file.read())
            tmp_file_path = tmp_file.name
        
        # Get audio file duration and process
        with sr.AudioFile(tmp_file_path) as source:
            duration = source.DURATION if hasattr(source, 'DURATION') else None
            st.info(f"Audio duration: {duration:.1f} seconds" if duration else "Processing audio file...")
            
            # Adjust recognizer settings for better accuracy
            recognizer.adjust_for_ambient_noise(source, duration=1)
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            
            # Record all audio data
            audio_data = recognizer.record(source)
        
        # Clean up temp file
        os.unlink(tmp_file_path)
        
        # Transcribe with extended timeout
        st.info("Transcribing audio...")
        transcript = recognizer.recognize_google(
            audio_data, 
            language='en-US',
            show_all=False
        )
        
        # Debug info
        word_count = len(transcript.split())
        char_count = len(transcript)
        st.info(f"Transcription complete: {word_count} words, {char_count} characters")
        
        return transcript, duration
        
    except sr.UnknownValueError:
        return "Could not understand the audio. Please ensure clear speech and minimal background noise.", None
    except sr.RequestError as e:
        return f"Google Speech Recognition service error: {str(e)}", None
    except Exception as e:
        return f"Transcription error: {str(e)}", None

def analyze_transcript(transcript, audio_duration=None):
    """Enhanced analysis with comprehensive extraction and debugging"""
    
    # Debug the transcript
    st.write(f"**Debug Info:** Transcript length: {len(transcript)} characters, {len(transcript.split())} words")
    
    # Enhanced attendee extraction with case-insensitive matching
    attendees = set()
    
    # More aggressive name extraction patterns
    name_patterns = [
        r'\b([A-Z][a-z]{2,})\s*:\s*',  # "John: said something"
        r'\b(?:I\'m|I am|This is|My name is|I\'ll be|speaking is)\s+([A-Z][a-z]{2,})',  
        r'\b([A-Z][a-z]{2,})\s+(?:said|mentioned|asked|stated|noted|added|replied|speaking|here)',  
        r'\b(?:Hi|Hello|Hey),?\s+(?:I\'m|I am|this is)\s+([A-Z][a-z]{2,})',  
        r'\b([A-Z][a-z]{2,})\s+(?:from|at|in|with)',  
        r'\b(?:with|and|from|by)\s+([A-Z][a-z]{2,})',
        r'\b([A-Z][a-z]{2,})\s+(?:will|would|should|can|could)',
        r'\bMr\.?\s+([A-Z][a-z]{2,})',
        r'\bMs\.?\s+([A-Z][a-z]{2,})',
        r'\b([A-Z][a-z]{2,})\s+(?:thinks|believes|suggests|recommends)',
    ]
    
    # Extract names more aggressively
    for pattern in name_patterns:
        matches = re.findall(pattern, transcript, re.IGNORECASE)
        for match in matches:
            # Filter out common words
            if (len(match) > 2 and match.lower() not in 
                ['the', 'and', 'this', 'that', 'with', 'from', 'said', 'will', 'can', 'may', 'should', 
                 'have', 'been', 'were', 'they', 'what', 'when', 'where', 'how', 'why', 'who', 'which']):
                attendees.add(match.title())  # Normalize case
    
    # Also look for any capitalized words that might be names
    words = transcript.split()
    for i, word in enumerate(words):
        if (word[0].isupper() and len(word) > 2 and 
            word.lower() not in ['the', 'and', 'this', 'that', 'with', 'from', 'said', 'will', 'can', 'may', 'should'] and
            not word.endswith(':') and not word.endswith('.') and not word.endswith(',')):
            # Check if it looks like a name (appears multiple times or is followed by action words)
            if (words.count(word) > 1 or 
                (i + 1 < len(words) and words[i + 1].lower() in ['said', 'mentioned', 'will', 'should', 'can'])):
                attendees.add(word)
    
    attendees_list = list(attendees) if attendees else ["Meeting Participants"]
    
    # More comprehensive action item extraction
    action_items = []
    
    # Split into sentences and process each one
    sentences = re.split(r'[.!?]+', transcript)
    
    # Much more comprehensive action patterns
    action_patterns = [
        # Direct assignments with names
        r'\b([A-Z][a-z]+)\s+(?:will|should|needs? to|must|has to|is going to|gonna)\s+(.{10,})',
        r'\b([A-Z][a-z]+),?\s+(?:can you|could you|please|you should|you need to|you must)\s+(.{10,})',
        r'\bassign(?:ed|ing)?\s+(?:to\s+)?([A-Z][a-z]+)\s+(?:to\s+)?(.{10,})',
        
        # General action statements
        r'\b(?:will|should|need to|must|have to|going to|gonna)\s+(.{10,})',
        r'\b(?:let\'s|we should|we need to|we must|we have to|we will)\s+(.{10,})',
        r'\b(?:someone|somebody)\s+(?:should|needs to|must|has to)\s+(.{10,})',
        
        # Task and follow-up items
        r'\baction\s+item[s]?:?\s*(.{10,})',
        r'\btodo[s]?:?\s*(.{10,})',
        r'\btask[s]?:?\s*(.{10,})',
        r'\bfollow\s+up\s+(?:on\s+|with\s+)?(.{10,})',
        r'\bnext\s+step[s]?\s+(?:is|are|will be|should be)\s+(.{10,})',
        
        # Deadline and time-based items
        r'\bby\s+(?:next\s+week|friday|monday|tuesday|wednesday|thursday|saturday|sunday|end of|deadline)\s+(.{10,})',
        r'\b(?:due|deadline|before|until)\s+(.{10,})',
        r'\bdeadline\s+(?:is|for)\s+(.{10,})',
        
        # Meeting outcomes and decisions
        r'\bdecided?\s+(?:to|that)\s+(.{10,})',
        r'\bagreed?\s+(?:to|that)\s+(.{10,})',
        r'\bresolved?\s+(?:to|that)\s+(.{10,})',
        
        # Questions that need answers
        r'\bneed[s]?\s+to\s+(?:find out|check|verify|confirm)\s+(.{10,})',
        r'\bwho\s+(?:will|should|can)\s+(.{10,})',
        r'\bwhen\s+(?:will|should|can)\s+(?:we|someone|somebody)\s+(.{10,})',
    ]
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:  # Skip very short sentences
            continue
            
        for pattern in action_patterns:
            matches = re.findall(pattern, sentence, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    assignee, task = match[0], match[1]
                elif isinstance(match, tuple) and len(match) == 1:
                    assignee = "Team Member"
                    task = match[0]
                else:
                    assignee = "Team Member"
                    task = match
                
                # Clean up the task
                task = task.strip()
                task = re.sub(r'^(?:to\s+|that\s+|and\s+|the\s+)', '', task, flags=re.IGNORECASE)
                task = re.sub(r'\s+', ' ', task)  # Clean multiple spaces
                
                # Only add substantial tasks
                if (len(task) > 15 and len(task) < 200 and 
                    task not in [item['task'] for item in action_items] and
                    not task.lower().startswith(('and', 'or', 'but', 'so', 'if', 'when', 'where', 'why', 'how'))):
                    
                    # Enhanced priority detection
                    priority = "Medium"
                    high_words = ['urgent', 'asap', 'immediately', 'critical', 'emergency', 'rush', 'important', 'priority', 'key', 'essential']
                    low_words = ['when possible', 'eventually', 'nice to have', 'optional', 'if time', 'later', 'someday']
                    
                    if any(word in task.lower() for word in high_words):
                        priority = "High"
                    elif any(word in task.lower() for word in low_words):
                        priority = "Low"
                    
                    # Enhanced due date detection
                    due_date = "To be determined"
                    if any(word in task.lower() for word in ['by friday', 'by monday', 'this week', 'next week']):
                        due_date = "This/Next week"
                    elif any(word in task.lower() for word in ['by end of', 'month', 'quarterly']):
                        due_date = "End of month/quarter"
                    elif any(word in task.lower() for word in ['asap', 'immediately', 'urgent', 'today', 'tomorrow']):
                        due_date = "ASAP"
                    elif any(word in task.lower() for word in ['deadline', 'due']):
                        due_date = "Check deadline"
                    
                    # Better assignee validation
                    if isinstance(assignee, str) and len(assignee) > 2:
                        if assignee in attendees_list:
                            final_assignee = assignee
                        else:
                            # Check if any attendee name is in the task
                            final_assignee = "Team Member"
                            for attendee in attendees_list:
                                if attendee.lower() in task.lower():
                                    final_assignee = attendee
                                    break
                            if final_assignee == "Team Member" and attendees_list and attendees_list[0] != "Meeting Participants":
                                final_assignee = attendees_list[0]
                    else:
                        final_assignee = attendees_list[0] if attendees_list and attendees_list[0] != "Meeting Participants" else "Team Member"
                    
                    action_items.append({
                        "task": task,
                        "assignee": final_assignee,
                        "due_date": due_date,
                        "priority": priority,
                        "status": "Pending"
                    })
    
    # Enhanced topic extraction
    topics = []
    
    # Look for explicit topic mentions
    topic_patterns = [
        r'(?:discuss|discussing|talk about|talking about|review|reviewing|cover|covering|address|addressing)\s+(.{15,})',
        r'(?:regarding|about|concerning|on the topic of|subject of|topic is|subject is)\s+(.{15,})',
        r'(?:agenda item|topic|subject|issue|matter|point)(?:\s+\d+)?:?\s*(.{15,})',
        r'(?:moving on to|next topic|another topic|also discussing|let\'s talk about)\s+(.{15,})',
        r'(?:the main|primary|key|important|central)\s+(?:topic|issue|point|concern|matter)\s+(?:is|was|here)\s+(.{15,})',
        r'(?:question|problem|challenge|issue)\s+(?:is|was|about|with|regarding)\s+(.{15,})',
        r'(?:focus|focusing)\s+(?:on|is|was)\s+(.{15,})',
        r'(?:update|status|report)\s+(?:on|about|regarding)\s+(.{15,})',
    ]
    
    for pattern in topic_patterns:
        matches = re.findall(pattern, transcript, re.IGNORECASE)
        for match in matches:
            topic = match.strip()
            topic = re.sub(r'^(?:is\s+|was\s+|that\s+|the\s+|a\s+|an\s+)', '', topic, flags=re.IGNORECASE)
            topic = re.sub(r'\s+', ' ', topic)
            
            if (len(topic) > 10 and len(topic) < 100 and 
                topic not in topics and
                not topic.lower().startswith(('and', 'or', 'but', 'so', 'if', 'when', 'where', 'why', 'how'))):
                topics.append(topic)
    
    # If still not enough topics, extract from meaningful sentences
    if len(topics) < 3:
        sentences = re.split(r'[.!?]+', transcript)
        for sentence in sentences:
            sentence = sentence.strip()
            if (len(sentence) > 25 and len(sentence) < 120 and
                not sentence.lower().startswith(('and', 'or', 'but', 'so', 'if', 'when', 'where', 'why', 'how')) and
                not re.match(r'^[A-Z][a-z]+\s*:', sentence)):  # Avoid "Name: said something"
                topics.append(sentence)
                if len(topics) >= 5:
                    break
    
    if not topics:
        topics = ["General Discussion"]
    
    # Calculate duration
    if audio_duration:
        duration_str = f"{int(audio_duration // 60)}:{int(audio_duration % 60):02d}"
    else:
        estimated_minutes = len(transcript.split()) / 150
        duration_str = f"~{int(estimated_minutes)} minutes"
    
    # Debug output
    st.write(f"**Analysis Results:** Found {len(attendees_list)} attendees, {len(action_items)} action items, {len(topics)} topics")
    
    return {
        "meeting_summary": {
            "title": "Meeting Notes",
            "date": datetime.now().strftime('%Y-%m-%d'),
            "estimated_duration": duration_str,
            "key_topics": topics[:5]
        },
        "attendees": attendees_list[:10],
        "detailed_notes": [
            {
                "topic": "Complete Meeting Discussion",
                "details": transcript,  # Full transcript preserved
                "owner": "All Participants"
            }
        ],
        "action_items": action_items
    }

def display_results(data, transcript):
    """Display organized results"""
    
    # Meeting Summary
    st.header("📋 Meeting Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Meeting Details")
        st.write(f"**Title:** {data['meeting_summary']['title']}")
        st.write(f"**Date:** {data['meeting_summary']['date']}")
        st.write(f"**Duration:** {data['meeting_summary']['estimated_duration']}")
    
    with col2:
        st.subheader("Attendees")
        for attendee in data['attendees']:
            st.write(f"• {attendee}")
    
    st.subheader("Key Topics Discussed")
    for i, topic in enumerate(data['meeting_summary']['key_topics'], 1):
        st.write(f"{i}. {topic}")
    
    # Action Items
    st.header("✅ Action Items")
    
    # Group by priority
    high_priority = [item for item in data['action_items'] if item['priority'] == 'High']
    medium_priority = [item for item in data['action_items'] if item['priority'] == 'Medium']
    low_priority = [item for item in data['action_items'] if item['priority'] == 'Low']
    
    for priority_name, items, emoji in [
        ("High Priority", high_priority, "🔴"),
        ("Medium Priority", medium_priority, "🟡"),
        ("Low Priority", low_priority, "🟢")
    ]:
        if items:
            st.subheader(f"{emoji} {priority_name}")
            for i, item in enumerate(items, 1):
                st.write(f"**{i}. {item['task']}**")
                st.write(f"   👤 Assignee: {item['assignee']}")
                st.write(f"   📅 Due: {item['due_date']}")
                st.write(f"   📊 Status: {item['status']}")
                st.write("")
    
    # Full Transcript
    st.header("📝 Full Transcript")
    with st.expander("View Complete Transcript", expanded=False):
        st.text_area("", transcript, height=300, disabled=True)
    
    # Download Report
    st.header("📥 Download Report")
    
    report = f"""
MEETING NOTES REPORT
===================

Meeting: {data['meeting_summary']['title']}
Date: {data['meeting_summary']['date']}
Duration: {data['meeting_summary']['estimated_duration']}

ATTENDEES
---------
{chr(10).join([f"• {attendee}" for attendee in data['attendees']])}

KEY TOPICS
----------
{chr(10).join([f"{i}. {topic}" for i, topic in enumerate(data['meeting_summary']['key_topics'], 1)])}

ACTION ITEMS
------------
{chr(10).join([f"{i}. {item['task']}{chr(10)}   Assignee: {item['assignee']}{chr(10)}   Due: {item['due_date']}{chr(10)}   Priority: {item['priority']}{chr(10)}" for i, item in enumerate(data['action_items'], 1)])}

FULL TRANSCRIPT
---------------
{transcript}

---
Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Processing: 100% FREE - No API costs
"""
    
    st.download_button(
        "📄 Download Complete Report",
        data=report,
        file_name=f"meeting_notes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Main App
def main():
    st.markdown("### 🎯 Instructions:")
    st.info("Upload your WAV file below and click 'Process Audio' - no additional setup required!")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Choose a WAV audio file",
        type=['wav'],
        help="This app works specifically with WAV files to avoid dependency issues"
    )
    
    if uploaded_file is not None:
        file_size_mb = uploaded_file.size / (1024 * 1024)
        
        # Display file info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("File Name", uploaded_file.name)
        with col2:
            st.metric("File Size", f"{file_size_mb:.1f} MB")
        with col3:
            st.metric("💰 Cost", "$0.00")
        
        # Process button
        if st.button("🚀 Process Audio", type="primary"):
            if file_size_mb > 25:
                st.warning("File is quite large. Processing may take longer.")
            
            start_time = time.time()
            
            # Transcribe
            with st.spinner("Processing audio file..."):
                result = transcribe_wav_file(uploaded_file)
                if isinstance(result, tuple):
                    transcript, audio_duration = result
                else:
                    transcript, audio_duration = result, None
            
            if transcript and len(transcript.strip()) > 10 and not transcript.startswith(("Could not", "Google Speech", "Transcription error")):
                transcription_time = time.time() - start_time
                
                # Analyze
                with st.spinner("Analyzing transcript and extracting insights..."):
                    results = analyze_transcript(transcript, audio_duration)
                
                total_time = time.time() - start_time
                
                # Success message
                st.success(f"✅ Processing completed in {total_time:.1f} seconds!")
                
                # Performance metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Transcription Time", f"{transcription_time:.1f}s")
                with col2:
                    st.metric("Total Time", f"{total_time:.1f}s")
                with col3:
                    st.metric("Words Transcribed", len(transcript.split()))
                with col4:
                    if audio_duration:
                        st.metric("Audio Duration", f"{int(audio_duration//60)}:{int(audio_duration%60):02d}")
                    else:
                        st.metric("Estimated Duration", results['meeting_summary']['estimated_duration'])
                
                st.markdown("---")
                
                # Display results
                display_results(results, transcript)
                
            else:
                st.error(f"❌ Transcription failed: {transcript}")
                st.info("""
                **Possible solutions:**
                • Ensure the audio is clear with minimal background noise
                • Check that the audio contains speech (not just music/silence)
                • Try converting the audio to a different WAV format
                • Ensure you have internet connection (required for Google Speech Recognition)
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("**💡 Tips for Best Results:**")
    st.markdown("• Use clear audio recordings with minimal background noise")
    st.markdown("• Ensure speakers are close to the microphone")
    st.markdown("• Internet connection required for speech recognition")
    st.markdown("• Processing time depends on audio length and internet speed")

if __name__ == "__main__":
    main()
