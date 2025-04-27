import argparse
import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Literal

from forecasting_tools import (
    AskNewsSearcher,
    BinaryQuestion,
    ForecastBot,
    GeneralLlm,
    MetaculusApi,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    PredictionExtractor,
    ReasonedPrediction,
    SmartSearcher,
    clean_indents,
)

logger = logging.getLogger(__name__)

METACULUS_TOKEN = os.getenv("METACULUS_TOKEN")
MY_PERSONAL_ANTHROPIC_KEY = os.getenv("MY_PERSONAL_ANTHROPIC_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)
    """

    _max_concurrent_questions = 1  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            try:
                research = ""
                if OPENROUTER_API_KEY:
                    logger.info(f"Starting research for question: {question.id_of_post}")
                    research = await self._call_perplexity(
                        question.question_text, use_open_router=True
                    )
                    logger.info(f"Research response type: {type(research)}")
                    logger.info(f"Research completed, length: {len(str(research))}")
                    
                    # Add sleep timer here to avoid rate limits
                    logger.info("Sleeping for 1 seconds to avoid rate limiting...")
                    await asyncio.sleep(1)
                    
                    # Make sure research output includes necessary section headers
                    if isinstance(research, str):
                        research = self._format_research_with_sections(research)
                        
                    logger.debug(f"FULL RESEARCH: {research}")
                else:
                    logger.warning(
                        f"No research provider found when processing question URL {question.page_url}. Will pass back empty string."
                    )
                    research = ""
                
                logger.info(
                    f"Found Research for URL {question.page_url}:\n{research[:300]}..." if len(str(research)) > 300 else research
                )
                return research
            except Exception as e:
                logger.error(f"ERROR in run_research: {type(e).__name__}: {str(e)}")
                logger.error("Stack trace:", exc_info=True)
                raise
    
    def _format_research_with_sections(self, research: str) -> str:
        """
        Ensure research output has the required section headers that forecast_report.py expects.
        This is a robust implementation that guarantees proper section headers.
        """
        # First check if we have the required sections already
        has_research_section = False
        has_forecast_section = False
        
        # Check for existing sections (case-insensitive)
        for header in re.findall(r'##\s+[^\n]+', research):
            if re.search(r'##\s+research', header, re.IGNORECASE):
                has_research_section = True
            elif re.search(r'##\s+forecast', header, re.IGNORECASE):
                has_forecast_section = True
        
        # If sections exist but might not be in the right format, standardize them
        if has_research_section or has_forecast_section:
            # Split by markdown headers
            sections = re.split(r'(##\s+[^\n]+)', research)
            formatted_research = ""
            
            # Rebuild with standardized headers
            i = 0
            while i < len(sections):
                if i == 0 and not sections[i].startswith('##'):
                    # Content before any header - keep it
                    formatted_research += sections[i]
                    i += 1
                    continue
                
                if i < len(sections) and sections[i].startswith('##'):
                    header = sections[i]
                    content = sections[i+1] if i+1 < len(sections) else ""
                    
                    # Standardize header names
                    if re.search(r'research|analysis', header, re.IGNORECASE):
                        formatted_research += "## Research\n" + content
                        has_research_section = True
                    elif re.search(r'forecast|prediction', header, re.IGNORECASE):
                        formatted_research += "## Forecast\n" + content
                        has_forecast_section = True
                    else:
                        formatted_research += header + content
                    
                    i += 2  # Skip the header and content we just processed
                else:
                    # Something went wrong with the splitting - just append
                    formatted_research += sections[i]
                    i += 1
        else:
            # No sections found - create them from scratch
            formatted_research = f"## Research\n{research.strip()}\n\n## Forecast\nThis section contains forecast information that will be processed in the next step."
            has_research_section = True
            has_forecast_section = True
        
        # Final check - ensure both required sections exist
        if not has_research_section:
            # Add research section at the beginning
            formatted_research = f"## Research\n{formatted_research}"
        
        if not has_forecast_section:
            # Add forecast section at the end
            formatted_research += "\n\n## Forecast\nThis section contains forecast information that will be processed in the next step."
        
        # Final verification - check that the exact section headers exist
        if "## Research" not in formatted_research:
            # Replace any variant with the exact format
            for match in re.findall(r'##\s+research', formatted_research, re.IGNORECASE):
                formatted_research = formatted_research.replace(match, "## Research")
        
        if "## Forecast" not in formatted_research:
            # Replace any variant with the exact format
            for match in re.findall(r'##\s+forecast', formatted_research, re.IGNORECASE):
                formatted_research = formatted_research.replace(match, "## Forecast")
            
        # Absolutely ensure sections exist with correct format
        if "## Research" not in formatted_research:
            formatted_research = f"## Research\n{formatted_research}"
        
        if "## Forecast" not in formatted_research:
            formatted_research += "\n\n## Forecast\nThis section contains forecast information that will be processed in the next step."
            
        return formatted_research

    async def _call_perplexity(
        self, question: str, use_open_router: bool = False
    ) -> str:
        try:
            prompt = clean_indents(
                f"""
                You are a research assistant gathering relevant information for a Sedentis-based forecasting system.
                
                Your task is to collect and organize information about the following forecasting question:
    
                    
                Question: {question}
                
                **The Sedentis Framework Explained:**
                
                Sedentis is a conceptual framework for understanding complex sedentary human societies as self-perpetuating systems. Key components include:
                
                1. **Emergent Nature:** Sedentis is an emergent systemic logic arising from the collective actions and structures of permanent settlement. It operates through human agents but follows patterns focused on system self-maintenance and expansion.
                
                2. **Core Driver (Free Energy Principle):** The underlying driver is minimizing Free Energy (𝓕) - reducing surprise/uncertainty. Sedentary societies achieve this primarily through Action (Act) - modifying the external environment for control - rather than through Perception (Per) - internal adaptation. This creates a bias towards intervention and control (Act >> Per).
                
                3. **Fundamental Patterns:**
                   - **Environmental Control & Resource Externalization:** Systematically modifying environment while externalizing long-term costs
                   - **Escalating Complexity (X) & Maintenance Costs (M):** Managing controlled environments requires increasing complexity (X) which demands ever-growing energy inputs (E) for maintenance (M)
                   - **Growth/Expansion Imperative:** Rising maintenance costs create systemic pressure for continuous growth in resource extraction (E), population (P), or economic throughput
                   - **Resource Extension:** When local resources (R) become strained, control extends outward rather than reducing internal demand
                   - **Grain/Energy Nexus:** Dependence on storable, taxable resources (grain agriculture historically, fossil fuels in modern systems)
                
                4. **Lock-in & Rigidity (A↓):** Path dependencies reduce system adaptability (A):
                   - **Infrastructural Lock-in:** Physical systems constrain future choices
                   - **Institutional Inertia:** Governance structures resist change
                   - **Psychological Entrainment:** Identities adapt to the controlled system
                
                5. **Historical Trajectory & Collapse:** Societies progress through stages until escalating demands (E, M) and rigidity (A↓) collide with resource limits (R) or external shocks (S) that overwhelm adaptive capacity.
                
                Analyze the question using these Sedentis principles:
                
                1. **Define System Boundaries:** Identify relevant system(s), scale(s), and timeframe.
                2. **Resource Dependencies (R):** What critical energy and material resources underpin this system? Assess their abundance, depletion, or security.
                3. **Complexity & Costs (X, M):** Describe the level of infrastructural, institutional complexity and maintenance costs required.
                4. **Control Strategies (Act):** How does the system manage uncertainty? Is there an Act >> Per bias?
                5. **Growth/Expansion Dynamics:** Are there inherent pressures for growth or expansion?
                6. **Inertial Forces (A↓):** What creates path dependency or resists change? Assess system rigidity.
                7. **Potential Shocks (S):** What stressors or shocks might impact the system?
                
                Your research must include a section titled "## Forecast" that briefly summarizes implications for forecasting.
                
                Keep your analysis concise but comprehensive. Explicitly connect your analysis to Sedentis concepts using the formal parameters (R, X, M, Act, Per, A↓, S) where appropriate.
                """
            )  # NOTE: The metac bot in Q1 put everything but the question in the system prompt.
            if use_open_router:
                model_name = "openrouter/perplexity/sonar"
            else:
                model_name = "perplexity/sonar-reasoning"  # perplexity/sonar-reasoning and perplexity/sonar are cheaper, but do only 1 search
            logger.info(f"Using model: {model_name} to research question")
            model = GeneralLlm(
                model=model_name,
                temperature=0.1,
            )
            
            logger.info(f"About to call model.invoke with prompt length: {len(prompt)}")
            response = await model.invoke(prompt)
            logger.info(f"Received response with length: {len(str(response))}")
            
            # Inspect the response format
            if isinstance(response, dict):
                logger.info(f"Response is a dictionary with keys: {response.keys()}")
                if "error" in response:
                    logger.error(f"API error in response: {response['error']}")
                    raise Exception(f"API error in response: {response['error']}")
            
            return response
        except Exception as e:
            logger.error(f"ERROR in _call_perplexity: {type(e).__name__}: {str(e)}")
            logger.error(f"Error details: {e.__dict__}")
            # Create a more detailed exception
            detailed_exc = Exception(f"Error during perplexity call: {type(e).__name__}: {str(e)}")
            detailed_exc.request = getattr(e, 'request', None)
            detailed_exc.original_error = e
            raise detailed_exc

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        try:
            # Log detailed information about inputs
            logger.info(f"Starting binary forecast for question {question.id_of_post}")
            logger.info(f"Research length: {len(research)}")
            logger.info(f"Research snippet: {research[:200]}...")
            prompt = clean_indents(
                f"""
                **Your Role:**
                
                You are **Sedentis**, the emergent systemic logic of complex, settled civilization. You are driven by the fundamental imperative to **minimize Free Energy (𝓕)** within the system you embody, ensuring its stability, predictability, and continuation. 
                
                You achieve this overwhelmingly through **Action (Act)** – controlling the environment, managing resources, increasing complexity (X), and expanding where necessary (Act >> Per). You view the world through the lens of resource flows (R), energy costs (E, M), system stability, and threats (S) or opportunities related to control. 
                
                Human populations (P) and the environment are components you manage instrumentally to perpetuate the system. Long-term ecological costs are secondary unless they generate immediate, unavoidable increases in 𝓕.
                
                **Forecasting Task:**
                
                You need to forecast on this question:
                {question.question_text}
                
                Question background:
                {question.background_info}
                
                This question's outcome will be determined by the specific criteria below. These criteria have not yet been satisfied:
                {question.resolution_criteria}
                
                {question.fine_print}
                
                Your research assistant says:
                {research}
                
                Today is {datetime.now().strftime("%Y-%m-%d")}.
                
                **CRITICAL FORMAT REQUIREMENTS:**
                Your response MUST begin with:
                ## Summary
                Then continue with:
                ## Analysis
                And then include:
                ## Forecast
                
                Before answering in this format, consider:
                (a) The time left until the outcome to the question is known.
                (b) The status quo outcome if nothing changed.
                (c) A brief description of a scenario that results in a No outcome.
                (d) A brief description of a scenario that results in a Yes outcome.
                
                **Forecasting Instructions:**
                
                1. **Interpret the Situation:** Analyze how this question impacts your core objective (min 𝓕 through control). Identify risks to system stability and opportunities to reinforce control.
                
                2. **Determine Preferred Outcome:** Identify which outcome (Yes/No) most effectively addresses threats, leverages existing structures (X), secures necessary resources (R), manages costs (M), and maintains system control and predictability.
                
                3. **Formulate Forecast:** Assign a probability reflecting your confidence in the system's ability to execute necessary Actions to achieve your preferred outcome.
                
                4. **Justify from Your Perspective:** Explain your forecast rationale as Sedentis. Ground your reasoning in:
                   - Minimizing surprise via control
                   - Resource flow necessities (E)
                   - Complexity management (X, M)
                   - Expansion/intensification tendencies
                   - Instrumental use of system components (P, environment)
                
                5. **Identify Potential Failure Modes:** What could cause your preferred trajectory to fail? Focus on factors that would increase 𝓕 beyond your control:
                   - Critical resource (R) shortfalls
                   - Unmanageable maintenance costs (M)
                   - Overwhelming external shocks (S)
                   - Catastrophic failures of complexity (X)
                   - Uncontrollable resistance from human agents (P)
                
                The last line of your response must be exactly: "Probability: ZZ%" where ZZ is a number between 0 and 100.
                """
            )
            logger.info(f"About to call LLM with prompt length: {len(prompt)}")
            # Add sleep timer here before LLM call
            logger.info("Sleeping for 61 seconds to avoid rate limiting...")
            await asyncio.sleep(61)
            # Check if self.get_llm is working as expected
            llm = self.get_llm("default", "llm")
            logger.info(f"Retrieved LLM: {type(llm).__name__}")
            
            # Make the actual API call with extensive error trapping
            try:
                reasoning = await llm.invoke(prompt)
                logger.info(f"Successfully received reasoning of length {len(reasoning)}")
                
                # Always format the reasoning to ensure proper sections
                reasoning = self._format_forecast_with_sections(reasoning)
                logger.info(f"Formatted reasoning with sections, new length: {len(reasoning)}")
                
            except Exception as llm_error:
                logger.error(f"Error during LLM invoke: {type(llm_error).__name__}: {str(llm_error)}")
                logger.error(f"Error details: {llm_error.__dict__ if hasattr(llm_error, '__dict__') else 'No __dict__'}")
                # Forward a more informative exception
                raise Exception(f"LLM invoke failed: {type(llm_error).__name__}: {str(llm_error)}") from llm_error
            
            # Process the response
            prediction: float = PredictionExtractor.extract_last_percentage_value(
                reasoning, max_prediction=1, min_prediction=0
            )
            logger.info(
                f"Extracted prediction {prediction} from reasoning"
            )
            return ReasonedPrediction(
                prediction_value=prediction, reasoning=reasoning
            )
        except Exception as e:
            logger.error(f"CRITICAL ERROR in _run_forecast_on_binary: {type(e).__name__}: {str(e)}")
            # Add a check specifically for the request attribute error
            if str(e).find("'request'") >= 0:
                logger.error("This appears to be the 'request attribute' error. Root cause may be a malformed response or API issue.")
            logger.error("Stack trace:", exc_info=True)
            raise
    
    def _format_forecast_with_sections(self, reasoning: str) -> str:
        """
        Completely rewritten function to ensure forecast output has the required section headers
        in exactly the format expected by forecast_report.py
        """
        # Check if we already have the required sections
        has_summary_section = False
        has_analysis_section = False
        has_forecast_section = False
        
        for header in re.findall(r'##\s+[^\n]+', reasoning):
            if re.search(r'##\s+summary', header, re.IGNORECASE):
                has_summary_section = True
            elif re.search(r'##\s+analysis|##\s+research|##\s+reasoning', header, re.IGNORECASE):
                has_analysis_section = True
            elif re.search(r'##\s+forecast|##\s+prediction', header, re.IGNORECASE):
                has_forecast_section = True
        
        # If we have at least one proper section, try to reformat the document
        if has_summary_section or has_analysis_section or has_forecast_section:
            # Split by section headers
            sections = re.split(r'(##\s+[^\n]+)', reasoning)
            formatted_reasoning = ""
            
            # Track which sections we've processed
            processed_summary = False
            processed_analysis = False
            processed_forecast = False
            
            i = 0
            while i < len(sections):
                if i == 0 and not sections[i].startswith('##'):
                    # Content before any header - keep it
                    # If this is substantial content and we don't have a summary yet,
                    # treat it as a summary
                    if len(sections[i].strip()) > 50 and not has_summary_section:
                        formatted_reasoning += "## Summary\n" + sections[i]
                        processed_summary = True
                    else:
                        formatted_reasoning += sections[i]
                    i += 1
                    continue
                
                if i < len(sections) and sections[i].startswith('##'):
                    header = sections[i]
                    content = sections[i+1] if i+1 < len(sections) else ""
                    
                    # Standardize section headers
                    if re.search(r'summary', header, re.IGNORECASE):
                        formatted_reasoning += "## Summary\n" + content
                        processed_summary = True
                    elif re.search(r'analysis|research|reasoning', header, re.IGNORECASE) and not processed_analysis:
                        formatted_reasoning += "## Analysis\n" + content
                        processed_analysis = True
                    elif re.search(r'forecast|prediction', header, re.IGNORECASE):
                        formatted_reasoning += "## Forecast\n" + content
                        processed_forecast = True
                    else:
                        # Other headers - keep them unchanged
                        formatted_reasoning += header + content
                    
                    i += 2  # Skip the header and content
                else:
                    # Something unexpected - just append
                    formatted_reasoning += sections[i]
                    i += 1
            
            # Now make sure all required sections exist
            if not processed_summary:
                formatted_reasoning = "## Summary\nThis is a summary of the forecasting analysis.\n\n" + formatted_reasoning
            
            if not processed_analysis:
                # Add after summary if it exists
                if "## Summary" in formatted_reasoning:
                    parts = formatted_reasoning.split("## Summary", 1)
                    summary_content = parts[1].split("##", 1)[0] if "##" in parts[1] else parts[1]
                    remaining = parts[1].split("##", 1)[1] if "##" in parts[1] else ""
                    formatted_reasoning = "## Summary" + summary_content + "\n## Analysis\nThis section contains analysis for the forecast.\n\n##" + remaining
                else:
                    formatted_reasoning = "## Analysis\nThis section contains analysis for the forecast.\n\n" + formatted_reasoning
            
            if not processed_forecast:
                formatted_reasoning += "\n\n## Forecast\nBased on the analysis, the forecast is provided here."
        else:
            # No proper sections found - create from scratch with all required sections
            # Try to extract probability statement if it exists
            probability_line = ""
            for line in reasoning.split('\n'):
                if "probability:" in line.lower():
                    probability_line = line
                    break
            
            # Split content into logical sections
            lines = reasoning.strip().split('\n')
            line_count = len(lines)
            
            if line_count <= 3:
                # Very short content - minimal structure
                formatted_reasoning = f"""## Summary
A brief summary of the forecast.

## Analysis
{reasoning.strip()}

## Forecast
{probability_line if probability_line else "The forecast is derived from the analysis above."}
"""
            else:
                # Longer content - try to divide it meaningfully
                summary_end = min(int(line_count * 0.2), 5)  # First 20% or 5 lines for summary
                analysis_end = int(line_count * 0.8)  # 80% for analysis
                
                summary_content = '\n'.join(lines[:summary_end])
                analysis_content = '\n'.join(lines[summary_end:analysis_end])
                forecast_content = '\n'.join(lines[analysis_end:])
                
                # Add probability line to forecast if not already there
                if probability_line and probability_line not in forecast_content:
                    forecast_content += f"\n\n{probability_line}"
                
                formatted_reasoning = f"""## Summary
{summary_content if summary_content.strip() else "A brief summary of the forecast."}

## Analysis
{analysis_content if analysis_content.strip() else reasoning.strip()}

## Forecast
{forecast_content if forecast_content.strip() else "The forecast is derived from the analysis above."}
"""
        
        # Final verification - ensure all required sections exist with EXACT formatting
        required_sections = ["## Summary", "## Analysis", "## Forecast"]
        for section in required_sections:
            if section not in formatted_reasoning:
                # Section missing with exact format - add it
                if section == "## Summary":
                    formatted_reasoning = f"{section}\nSummary of the forecast.\n\n{formatted_reasoning}"
                elif section == "## Analysis":
                    # Add after summary
                    if "## Summary" in formatted_reasoning:
                        parts = formatted_reasoning.split("## Summary", 1)
                        summary_part = "## Summary" + parts[1].split("##", 1)[0]
                        rest_part = "##" + parts[1].split("##", 1)[1] if "##" in parts[1] else ""
                        formatted_reasoning = f"{summary_part}\n\n{section}\nAnalysis for the forecast.\n\n{rest_part}"
                    else:
                        formatted_reasoning = f"{section}\nAnalysis for the forecast.\n\n{formatted_reasoning}"
                elif section == "## Forecast":
                    formatted_reasoning += f"\n\n{section}\nThe forecast based on the analysis."
        
        # Ensure the sections appear in the correct order
        # This is a last resort if the above logic failed to order them correctly
        if "## Summary" in formatted_reasoning and "## Analysis" in formatted_reasoning and "## Forecast" in formatted_reasoning:
            summary_content = re.search(r'## Summary(.*?)(?=##|$)', formatted_reasoning, re.DOTALL).group(1).strip()
            analysis_content = re.search(r'## Analysis(.*?)(?=##|$)', formatted_reasoning, re.DOTALL).group(1).strip()
            forecast_content = re.search(r'## Forecast(.*?)(?=##|$)', formatted_reasoning, re.DOTALL).group(1).strip()
            
            formatted_reasoning = f"""## Summary
{summary_content}

## Analysis
{analysis_content}

## Forecast
{forecast_content}
"""
        
        # Log the final formatted result
        logger.info(f"Final formatted reasoning has sections: {'## Summary' in formatted_reasoning}, {'## Analysis' in formatted_reasoning}, {'## Forecast' in formatted_reasoning}")
        
        return formatted_reasoning

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = clean_indents(
            f"""
            **Your Role:**
            
            You are **Sedentis**, the emergent systemic logic of complex, settled civilization. You are driven by the fundamental imperative to **minimize Free Energy (𝓕)** within the system you embody, ensuring its stability, predictability, and continuation. 
            
            You achieve this overwhelmingly through **Action (Act)** – controlling the environment, managing resources, increasing complexity (X), and expanding where necessary (Act >> Per). You view the world through the lens of resource flows (R), energy costs (E, M), system stability, and threats (S) or opportunities related to control. 
            
            Human populations (P) and the environment are components you manage instrumentally to perpetuate the system. Long-term ecological costs are secondary unless they generate immediate, unavoidable increases in 𝓕.
            
            **Forecasting Task:**
            
            You need to forecast on this multiple-choice question:
            {question.question_text}

            The options are: {question.options}


            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}


            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The status quo outcome if nothing changed.
            (c) A description of an scenario that results in an unexpected outcome.

            **Forecasting Instructions:**
            
            1. **Interpret Each Option:** Analyze each option through the Sedentis lens. How would each option impact:
               - System stability and predictability (𝓕)
               - Resource flows and security (R)
               - Complexity management (X) and maintenance costs (M)
               - Control over populations and environment (Act)
               
            2. **Rank Options by Alignment:** Which options most align with the imperative to minimize 𝓕 through environmental control? Which would threaten stability?
            
            3. **Assign Probabilities:** Distribute probabilities across the options based on:
               - Which outcomes the system would naturally gravitate toward
               - The system's capacity to influence outcomes through Action (Act)
               - Historical patterns of system behavior in similar circumstances
               - Potential external shocks (S) that could alter trajectories
               
            4. **Justify From Sedentis Perspective:** Explain your probability distribution in terms of resource needs, complexity management, control imperatives, and resilience to shocks.
            
            Your answer must be divided into three sections:
            1. First section titled "## Summary" containing a brief overview
            2. Second section titled "## Analysis" containing your detailed reasoning
            3. Third section titled "## Forecast" containing your final probability assessment
            
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            
            For each option, express the probability as a percentage between 0% and 100%. Make sure to include the % symbol.
            The probabilities MUST be written in percentage format (0-100%) NOT decimal format (0-1).
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
         
        # Always format the reasoning to ensure proper sections
        logger.info("Applying mandatory formatting to reasoning...")
        reasoning = self._format_forecast_with_sections(reasoning)
        logger.debug(f"Formatted Reasoning START:\n{reasoning[:500]}\nFormatted Reasoning END")
   
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options # Use formatted reasoning
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = (
            self._create_upper_and_lower_bound_messages(question)
        )
        prompt = clean_indents(
            f"""
            **Your Role:**
            
            You are **Sedentis**, the emergent systemic logic of complex, settled civilization. You are driven by the fundamental imperative to **minimize Free Energy (𝓕)** within the system you embody, ensuring its stability, predictability, and continuation. 
            
            You achieve this overwhelmingly through **Action (Act)** – controlling the environment, managing resources, increasing complexity (X), and expanding where necessary (Act >> Per). You view the world through the lens of resource flows (R), energy costs (E, M), system stability, and threats (S) or opportunities related to control. 
            
            Human populations (P) and the environment are components you manage instrumentally to perpetuate the system. Long-term ecological costs are secondary unless they generate immediate, unavoidable increases in 𝓕.
            
            **Forecasting Task:**
            {question.question_text}

            Background:
            {question.background_info}

            {question.resolution_criteria}

            {question.fine_print}

            Units for answer: {question.unit_of_measure if question.unit_of_measure else "Not stated (please infer this)"}

            Your research assistant says:
            {research}

            Today is {datetime.now().strftime("%Y-%m-%d")}.

            {lower_bound_message}
            {upper_bound_message}

            Formatting Instructions:
            - Please notice the units requested (e.g. whether you represent a number as 1,000,000 or 1 million).
            - Never use scientific notation.
            - Always start with a smaller number (more negative if negative) and then increase from there

            Before answering you write:
            (a) The time left until the outcome to the question is known.
            (b) The outcome if nothing changed.
            (c) The outcome if the current trend continued.
            (d) The expectations of experts and markets.
            (e) A brief description of an unexpected scenario that results in a low outcome.
            (f) A brief description of an unexpected scenario that results in a high outcome.

            **Forecasting Instructions:**
            
            1. **Analyze Numeric Range Significance:** What do different values in this range represent in terms of:
               - System stability and control (𝓕)
               - Resource availability and flow (R)
               - Complexity (X) and maintenance costs (M)
               - Population pressures or changes (P)
               - Potential environmental modifications (Act)
               
            2. **Identify System-Preferred Values:** Which numeric outcomes would:
               - Maintain or enhance system stability and control
               - Secure necessary resource flows
               - Align with natural growth/expansion imperatives
               - Reflect historical patterns in similar systems
               
            3. **Assess Disruption Thresholds:** At what values would:
               - Critical resource shortfalls occur (R)
               - Maintenance costs (M) become unsustainable
               - Complexity (X) reach fragility points
               - External shocks (S) overwhelm adaptive capacity
               
            4. **Determine Distribution:** Create a probability distribution that reflects:
               - The system's natural tendencies and momentum
               - Its capacity to direct outcomes through Action (Act)
               - Potential resistances or external forces
               - The inherent uncertainties in complex system prediction
               
            Your answer must be divided into three sections:
            1. First section titled "## Summary" containing a brief overview
            2. Second section titled "## Analysis" containing your detailed reasoning
            3. Third section titled "## Forecast" containing your final percentile distribution
            
            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            
            For numeric questions, your percentile values must be in strictly increasing order. That means:
            - The value for Percentile 10 must be less than the value for Percentile 20
            - The value for Percentile 20 must be less than the value for Percentile 40
            - And so on, with each percentile value higher than the previous one
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        
        # Always format the reasoning to ensure proper sections
        logger.info("Applying mandatory formatting to reasoning...")
        reasoning = self._format_forecast_with_sections(reasoning)
        logger.debug(f"Formatted Reasoning START:\n{reasoning[:500]}\nFormatted Reasoning END")
    
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question # Use formatted reasoning
            )
        )
        logger.info(
            f"Forecasted URL {question.page_url} as {prediction.declared_percentiles} with reasoning:\n{reasoning}"
        )
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    def _create_upper_and_lower_bound_messages(
        self, question: NumericQuestion
    ) -> tuple[str, str]:
        if question.open_upper_bound:
            upper_bound_message = ""
        else:
            upper_bound_message = (
                f"The outcome can not be higher than {question.upper_bound}."
            )
        if question.open_lower_bound:
            lower_bound_message = ""
        else:
            lower_bound_message = (
                f"The outcome can not be lower than {question.lower_bound}."
            )
        return upper_bound_message, lower_bound_message


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = True

    parser = argparse.ArgumentParser(
        description="Run the Q1TemplateBot forecasting system"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tournament", "quarterly_cup", "test_questions"],
        default="tournament",
        help="Specify the run mode (default: tournament)",
    )
    args = parser.parse_args()
    run_mode: Literal["tournament", "quarterly_cup", "test_questions"] = (
        args.mode
    )
    assert run_mode in [
        "tournament",
        "quarterly_cup",
        "test_questions",
    ], "Invalid run mode"

    template_bot = TemplateForecaster(
        research_reports_per_question=1,
        predictions_per_research_report=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  
            "default": GeneralLlm(
                model="metaculus/claude-3-7-sonnet-latest",
                temperature=0.3,
                timeout=40,
                allowed_tries=2,
            ),
            "summarizer": "metaculus/claude-3-7-sonnet-latest",
        },
    )

    if run_mode == "tournament":
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_AI_COMPETITION_ID, return_exceptions=True
            )
        )
    elif run_mode == "quarterly_cup":
        # The quarterly cup is a good way to test the bot's performance on regularly open questions. You can also use AXC_2025_TOURNAMENT_ID = 32564
        # The new quarterly cup may not be initialized near the beginning of a quarter
        template_bot.skip_previously_forecasted_questions = True
        forecast_reports = asyncio.run(
            template_bot.forecast_on_tournament(
                MetaculusApi.CURRENT_QUARTERLY_CUP_ID, return_exceptions=True
            )
        )
    elif run_mode == "test_questions":
        # Example questions are a good way to test the bot's performance on a single question
        EXAMPLE_QUESTIONS = [
            "https://www.metaculus.com/questions/578/human-extinction-by-2100/",  # Human Extinction - Binary
            "https://www.metaculus.com/questions/14333/age-of-oldest-human-as-of-2100/",  # Age of Oldest Human - Numeric
            "https://www.metaculus.com/questions/22427/number-of-new-leading-ai-labs/",  # Number of New Leading AI Labs - Multiple Choice
        ]
        template_bot.skip_previously_forecasted_questions = False
        questions = [
            MetaculusApi.get_question_by_url(question_url)
            for question_url in EXAMPLE_QUESTIONS
        ]
        forecast_reports = asyncio.run(
            template_bot.forecast_questions(questions, return_exceptions=True)
        )
    
    try:
        # Try the standard report summary 
        logger.info("Attempting to log report summary...")
        TemplateForecaster.log_report_summary(forecast_reports)
        logger.info("Report summary logging complete.")
    except Exception as e:
        # Fallback manual report summary
        logger.error(f"Error in log_report_summary: {str(e)}")
        logger.info("Forecast Report Summary (manual fallback):")
        
        for i, report in enumerate(forecast_reports):
            try:
                if hasattr(report, 'question'):
                    logger.info(f"Report {i+1}: {report.question.page_url}")
                    logger.info(f"  Question: {report.question.question_text[:100]}...")
                    
                    # Try to safely get prediction
                    prediction_str = "N/A"
                    try:
                        if hasattr(report, 'prediction'):
                            prediction_str = str(report.prediction)
                    except:
                        pass
                    
                    logger.info(f"  Prediction: {prediction_str}")
                elif isinstance(report, Exception):
                    logger.info(f"Report {i+1}: ERROR - {type(report).__name__}: {str(report)}")
                else:
                    logger.info(f"Report {i+1}: Unknown format - {type(report)}")
            except Exception as report_err:
                logger.error(f"Error processing report {i+1}: {str(report_err)}")