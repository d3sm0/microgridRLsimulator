def _test_db():
    start_date = "2016-01-01"
    end_date = "2016-01-31"
    db = load_db(start_date, end_date)
    load = db.get('EPV', 10)
    print(load)
    slice_ = db.get('EPV', slice(0, 10))
    len(slice_)


if __name__ == "__main__":
    _test_db()
