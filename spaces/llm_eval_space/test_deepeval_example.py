from deepeval.synthesizer import Synthesizer
from deepeval.synthesizer.config import StylingConfig

styling_config = StylingConfig(
  input_format="Questions in English that asks for data in database.",
  expected_output_format="SQL query based on the given input",
  task="Answering text-to-SQL-related queries by querying a database and returning the results to users",
  scenario="Non-technical users trying to query a database using plain English.",
)

synthesizer = Synthesizer(styling_config=styling_config)

synthesizer.generate_goldens_from_scratch(num_goldens=5)
print(synthesizer.synthetic_goldens)

synthesizer.save_as(
    file_type='json',
    directory="./synthetic_data"
)

