def query_result_to_dict(query_result):
    result_list = []
    for row in query_result:
        row_dict = dict(row.__dict__)
        row_dict.pop('_sa_instance_state', None)  # Remove SQLAlchemy state
        result_list.append(row_dict)
    return result_list
