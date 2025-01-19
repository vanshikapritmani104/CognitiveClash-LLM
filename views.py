from django.shortcuts import render
from django.http import JsonResponse
from django.views.generic import View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.decorators.csrf import csrf_protect
from django.utils.decorators import method_decorator
import json
import logging
import os  # Import os for environment variables
from .models import Debate, Topic, DebateRound, Participant, DebateArgument
from django.core.exceptions import ObjectDoesNotExist
from .utils.crawler import DebateCrawler  # Import the DebateCrawler

# Initialize logger
logger = logging.getLogger(__name__)

# URLs to be crawled for gathering debate-related content
CRAWLER_URLS = [
    'https://example.com/debates',
    'https://example-forum.com/arguments'
]

class AIDebateView(LoginRequiredMixin, View):
    """Main view for the AI vs AI debate interface"""
    template_name = 'debates/ai_vs_ai.html'
    
    def get(self, request, debate_id):
        try:
            debate = Debate.objects.get(pk=debate_id)
            return render(request, self.template_name, {
                'debate_topic': debate.topic.title,
                'debate_id': debate_id
            })
        except Debate.DoesNotExist:
            logger.warning(f"Debate with ID '{debate_id}' not found.")
            return render(request, 'debates/error.html', {
                'error': 'Debate not found'
            })

@method_decorator(csrf_protect, name='dispatch')
class GenerateArgumentAPIView(LoginRequiredMixin, View):
    """API endpoint for generating debate arguments"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            debate_topic = data.get('prompt')
            stance = data.get('stance')
            turn = data.get('turn', 0)
            
            # Check if any related debates exist and crawl if not
            if not Debate.objects.filter(topic__title__icontains=debate_topic).exists():
                logger.info(f"No existing debates found for topic '{debate_topic}', initiating web crawl.")
                crawler = DebateCrawler(CRAWLER_URLS)
                crawler.crawl()  # Collect debate-related data

            # Retrieve relevant arguments from the database
            relevant_debates = Debate.objects.filter(topic__title__icontains=debate_topic)
            context_data = " ".join([debate.content for debate in relevant_debates[:5]])  # Use top 5 debates for context

            # Determine which LLM to use based on turn
            participant = Participant.objects.filter(
                llm_type='gemini' if turn % 2 == 0 else 'groq'
            ).first()
            
            if not participant:
                logger.error('No participants configured for LLM.')
                return JsonResponse({
                    'error': 'No participants configured'
                }, status=400)
            
            # Initialize appropriate LLM
            if participant.llm_type == 'gemini':
                llm = GeminiLLM(participant.api_key)
            else:
                llm = GroqLLM(participant.api_key)
            
            # Generate argument with specific stance and additional context
            prompt = (
                f"You are participating in a debate {stance} the topic: {debate_topic}. "
                f"Consider the following relevant arguments: {context_data}. "
                f"Generate a compelling argument for your position. "
                f"This is turn {turn} of the debate."
            )
            
            response = llm.generate_response(prompt)
            
            # Save the argument to the database
            try:
                debate = Debate.objects.get(topic__title=debate_topic)
            except ObjectDoesNotExist:
                logger.warning(f"Debate with topic '{debate_topic}' not found.")
                return JsonResponse({'error': 'Debate not found'}, status=404)
            
            current_round = debate.get_current_round()
            if current_round:
                current_round.arguments.create(
                    participant=participant,
                    content=response,
                    stance=stance
                )
            
            return JsonResponse({
                'argument': response
            })
            
        except json.JSONDecodeError:
            logger.error("JSON decoding failed in GenerateArgumentAPIView.")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            logger.error(f"Error generating argument for topic '{debate_topic}' with stance '{stance}': {str(e)}")
            return JsonResponse({'error': 'Failed to generate argument'}, status=500)

@method_decorator(csrf_protect, name='dispatch')
class GenerateSummaryAPIView(LoginRequiredMixin, View):
    """API endpoint for generating debate summaries"""
    
    def post(self, request):
        try:
            data = json.loads(request.body)
            debate_topic = data.get('prompt')
            debate_transcript = data.get('debate_transcript')
            
            # Use Gemini for summary generation
            llm = GeminiLLM(os.getenv('GEMINI_API_KEY'))
            
            # Include data collected by the web crawler in the summary context
            relevant_debates = Debate.objects.filter(topic__title__icontains=debate_topic)
            context_data = " ".join([debate.content for debate in relevant_debates[:5]])  # Use top 5 debates for context

            prompt = (
                f"Summarize the following debate on the topic '{debate_topic}'. "
                f"Analyze the key arguments made by both sides and provide a balanced assessment.\n\n"
                f"Debate transcript:\n{debate_transcript}\n\n"
                f"Relevant arguments collected: {context_data}"
            )
            
            summary = llm.generate_response(prompt)
            
            # Save summary to database
            try:
                debate = Debate.objects.get(topic__title=debate_topic)
            except ObjectDoesNotExist:
                logger.warning(f"Debate with topic '{debate_topic}' not found.")
                return JsonResponse({'error': 'Debate not found'}, status=404)
            
            debate.summary = summary
            debate.is_complete = True
            debate.save()
            
            return JsonResponse({
                'summary': summary
            })
            
        except json.JSONDecodeError:
            logger.error("JSON decoding failed in GenerateSummaryAPIView.")
            return JsonResponse({'error': 'Invalid JSON format'}, status=400)
        except Exception as e:
            logger.error(f"Error generating summary for topic '{debate_topic}': {str(e)}")
            return JsonResponse({'error': 'Failed to generate summary'}, status=500)