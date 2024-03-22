from django.db.models import Case, When


def get_objects_preserving_order(model, filter_by_arr, filter_field):
    queryset = model.objects.filter(**{f'{filter_field}__in': filter_by_arr})
    preserved_order = Case(*[When(**{filter_field: x, 'then': pos})
                             for pos, x in enumerate(filter_by_arr)])

    queryset_ordered = queryset.order_by(preserved_order)
    queryset_dict = {getattr(obj, filter_field): obj
                     for obj in list(queryset_ordered)}

    return [queryset_dict.get(x) for x in filter_by_arr]