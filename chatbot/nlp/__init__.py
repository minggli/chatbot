import pip


def ifninstall(pkg_name):
    """pip install language models used by spacy."""
    egg = '#egg='
    installed_pkg = [i.project_name for i in pip.get_installed_distributions()]
    egg_name = (egg in pkg_name) and pkg_name.split(egg)[1].replace('_', '-')
    if pkg_name and egg_name not in installed_pkg:
        pip.main(['install', pkg_name])


class _BaseNLP(object):
    sm_pkg = "https://github.com/explosion/spacy-models/releases/download/" \
             "en_core_web_sm-1.2.0/en_core_web_sm-1.2.0.tar.gz" \
             "#egg=en_core_web_sm"
    md_pkg = "https://github.com/explosion/spacy-models/releases/download/" \
             "en_core_web_md-1.2.1/en_core_web_md-1.2.1.tar.gz" \
             "#egg=en_core_web_md"
