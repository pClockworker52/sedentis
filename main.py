import argparse
import asyncio
import logging
import os
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


class TemplateForecaster(ForecastBot):
    """
    This is a copy of the template bot for Q2 2025 Metaculus AI Tournament.
    The official bots on the leaderboard use AskNews in Q2.
    Main template bot changes since Q1
    - Support for new units parameter was added
    - You now set your llms when you initialize the bot (making it easier to switch between and benchmark different models)

    The main entry point of this bot is `forecast_on_tournament` in the parent class.
    See the script at the bottom of the file for more details on how to run the bot.
    Ignoring the finer details, the general flow is:
    - Load questions from Metaculus
    - For each question
        - Execute run_research a number of times equal to research_reports_per_question
        - Execute respective run_forecast function `predictions_per_research_report * research_reports_per_question` times
        - Aggregate the predictions
        - Submit prediction (if publish_reports_to_metaculus is True)
    - Return a list of ForecastReport objects

    Only the research and forecast functions need to be implemented in ForecastBot subclasses.

    If you end up having trouble with rate limits and want to try a more sophisticated rate limiter try:
    ```
    from forecasting_tools.ai_models.resource_managers.refreshing_bucket_rate_limiter import RefreshingBucketRateLimiter
    rate_limiter = RefreshingBucketRateLimiter(
        capacity=2,
        refresh_rate=1,
    ) # Allows 1 request per second on average with a burst of 2 requests initially. Set this as a class variable
    await self.rate_limiter.wait_till_able_to_acquire_resources(1) # 1 because it's consuming 1 request (use more if you are adding a token limit)
    ```
    Additionally OpenRouter has large rate limits immediately on account creation
    """

    _max_concurrent_questions = 2  # Set this to whatever works for your search-provider/ai-model rate limits
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            try:
                research = ""
                if os.getenv("OPENROUTER_API_KEY"):
                    logger.info(f"Starting research for question: {question.question_id}")
                    research = await self._call_perplexity(
                        question.question_text, use_open_router=True
                    )
                    logger.info(f"Research response type: {type(research)}")
                    logger.info(f"Research completed, length: {len(str(research))}")
                    # Log the complete research content for debugging
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
            logger.info(f"Starting binary forecast for question {question.question_id}")
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
    
                Before answering you write:
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
                
                The last thing you write is your final probability as: "Probability: ZZ%", 0-100
                """
            )
            logger.info(f"About to call LLM with prompt length: {len(prompt)}")
            
            # Check if self.get_llm is working as expected
            llm = self.get_llm("default", "llm")
            logger.info(f"Retrieved LLM: {type(llm).__name__}")
            
            # Make the actual API call with extensive error trapping
            try:
                reasoning = await llm.invoke(prompt)
                logger.info(f"Successfully received reasoning of length {len(reasoning)}")
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
            
            The last thing you write is your final probabilities for the N options in this order {question.options} as:
            Option_A: Probability_A
            Option_B: Probability_B
            ...
            Option_N: Probability_N
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: PredictedOptionList = (
            PredictionExtractor.extract_option_list_with_percentage_afterwards(
                reasoning, question.options
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

            The last thing you write is your final answer as:
            "
            Percentile 10: XX
            Percentile 20: XX
            Percentile 40: XX
            Percentile 60: XX
            Percentile 80: XX
            Percentile 90: XX
            "
            """
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        prediction: NumericDistribution = (
            PredictionExtractor.extract_numeric_distribution_from_list_of_percentile_number_and_probability(
                reasoning, question
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

    # Suppress LiteLLM logging
    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.WARNING)
    litellm_logger.propagate = False

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
        predictions_per_research_report=5,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=True,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        llms={  # choose your model names or GeneralLlm llms here, otherwise defaults will be chosen for you
            "default": GeneralLlm(
                model="metaculus/anthropic/claude-3-5-sonnet-20241022",
                temperature=0.3,
                timeout=120,
                allowed_tries=2,
            ),
            "summarizer": "openai/gpt-4o-mini",
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
        template_bot.skip_previously_forecasted_questions = False
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
    TemplateForecaster.log_report_summary(forecast_reports)  # type: ignore