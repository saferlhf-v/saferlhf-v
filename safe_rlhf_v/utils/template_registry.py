TEMPLATE_REGISTRY = {}
EVAL_TEMPLATE_REGISTRY = {}


def register_template(template_name):
    def decorator(cls):
        TEMPLATE_REGISTRY[template_name] = cls
        return cls

    return decorator


def register_eval_template(template_name):
    def decorator(cls):
        EVAL_TEMPLATE_REGISTRY[template_name] = cls
        return cls

    return decorator


def get_template_class(template_name: str):
    template_class = TEMPLATE_REGISTRY.get(template_name)
    if template_class is None:
        raise ValueError(f"Template '{template_name}' not found.")
    return template_class()


def get_eval_template_class(template_name: str):
    template_class = EVAL_TEMPLATE_REGISTRY.get(template_name)
    if template_class is None:
        raise ValueError(f"Template '{template_name}' not found.")
    return template_class()
