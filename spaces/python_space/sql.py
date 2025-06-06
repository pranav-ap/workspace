import sqlalchemy as db

engine = db.create_engine("sqlite:///european_database.sqlite")
conn = engine.connect()

metadata = db.MetaData()
division = db.Table('divisions', metadata, autoload_with=engine)

print(repr(metadata.tables['divisions']))
print(division.columns.keys())

query = division.select() #SELECT * FROM divisions
print(query)

