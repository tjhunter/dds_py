# This file is copied and adapted from koalas's config system.

#
# Copyright (C) 2019 Databricks, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""
Infrastructure of options for Koalas.
"""
from typing import Union, Any, Tuple, Callable, List, Dict

from .structures import DDSException, DDSErrorCode


class Option:
    """
    Option class that defines an option with related properties.

    This class holds all information relevant to the one option. Also,
    Its instance can validate if the given value is acceptable or not.

    It is currently for internal usage only.

    Parameters
    ----------
    key: str, keyword-only argument
        the option name to use.
    doc: str, keyword-only argument
        the documentation for the current option.
    default: Any, keyword-only argument
        default value for this option.
    types: Union[Tuple[type, ...], type], keyword-only argument
        default is str. It defines the expected types for this option. It is
        used with `isinstance` to validate the given value to this option.
    check_func: Tuple[Callable[[Any], bool], str], keyword-only argument
        default is a function that always returns `True` with a empty string.
        It defines:
          - a function to check the given value to this option
          - the error message to show when this check is failed
        When new value is set to this option, this function is called to check
        if the given value is valid.

    Examples
    --------
    >>> option = Option(
    ...     key='option.name',
    ...     doc="this is a test option",
    ...     default="default",
    ...     types=(float, int),
    ...     check_func=(lambda v: v > 0, "should be a positive float"))

    >>> option.validate('abc')  # doctest: +NORMALIZE_WHITESPACE
    Traceback (most recent call last):
      ...
    ValueError: The value for option 'option.name' was <class 'str'>;
    however, expected types are [(<class 'float'>, <class 'int'>)].

    >>> option.validate(-1.1)
    Traceback (most recent call last):
      ...
    ValueError: should be a positive float

    >>> option.validate(1.1)
    """

    def __init__(
        self,
        *,
        key: str,
        doc: str,
        default: Any,
        types: Union[Tuple[type, ...], type] = str,
        check_func: Tuple[Callable[[Any], bool], str] = (lambda v: True, "")
    ):
        self.key = key
        self.doc = doc
        self.default = default
        self.types = types
        self.check_func = check_func

    def validate(self, v: Any) -> None:
        """
        Validate the given value and throw an exception with related information such as key.
        """
        if not isinstance(v, self.types):
            raise ValueError(
                "The value for option '%s' was %s; however, expected types are "
                "[%s]." % (self.key, type(v), str(self.types))
            )
        if not self.check_func[0](v):
            raise ValueError(self.check_func[1])


# Available options.
#

accept_list_option = Option(
    key="accept_list",
    doc=(
        "Accepts lists as objects. If true, lists are then traversed and their content is included in the signature"
        " (default true)"
    ),
    default=True,
    types=(bool,),
    check_func=(lambda v: True, "",),
)

accept_dict_option = Option(
    key="accept_dict",
    doc=(
        "Accepts dictionaries as objects. If true, lists are then traversed and their content is included "
        "in the signature"
        " (default true)"
    ),
    default=True,
    types=(bool,),
    check_func=(lambda v: True, "",),
)

extra_debug_option = Option(
    key="extra_debug",
    doc=(
        "Prints and evaluates extra debugging information. This information requires extra roundtrips to the "
        "storage backend. It is disabled by default to assist with debugging, but it can be disabled if "
        "I/O with the storage backend is an issue."
    ),
    default=True,
    types=(bool,),
    check_func=(lambda v: True, "",),
)

_options: List[Option] = [
    extra_debug_option,
    accept_list_option,
    accept_dict_option,
    Option(
        key="hash.max_sequence_size",
        doc=(
            "This sets a maximum size to a sequence (list, dictionary, tuple) that can be processed by DDS. "
            "Larger sequences are considered 'big objects' and should be loaded through functions. "
        ),
        default=10000,
        types=(int, type(None)),
        check_func=(
            lambda v: v is None or v >= 0,
            "'display.max_rows' should be greater than or equal to 0.",
        ),
    ),
]

_options_dict: Dict[str, Option] = dict(
    zip((option.key for option in _options), _options)
)
_options_values: Dict[str, Any] = dict(
    ((option.key, option.default) for option in _options)
)


def show_options():
    """
    Make a pretty table that can be copied and pasted into public documentation.
    This is currently for an internal purpose.
    """

    import textwrap

    header = ["Option", "Default", "Description"]
    row_format = "{:<31} {:<14} {:<53}"

    print(row_format.format("=" * 31, "=" * 14, "=" * 53))
    print(row_format.format(*header))
    print(row_format.format("=" * 31, "=" * 14, "=" * 53))

    for option in _options:
        doc = textwrap.fill(option.doc, 53)
        formatted = "".join(
            [line + "\n" + (" " * 47) for line in doc.split("\n")]
        ).rstrip()
        print(row_format.format(option.key, repr(option.default), formatted))

    print(row_format.format("=" * 31, "=" * 14, "=" * 53))


def get_option(key: Union[Option, str], default: Union[Any, None] = None) -> Any:
    """
    Retrieves the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    default : object
        The default value if the option is not set yet.

    Returns
    -------
    result : the value of the option

    Raises
    ------
    DDSException : if no such option exists and the default is not provided
    """
    if isinstance(key, Option):
        return get_option(key.key)
    _check_option(key)
    if default is None:
        default = _options_dict[key].default
    _options_dict[key].validate(default)

    return _options_values[key]


def set_option(key: str, value: Any) -> None:
    """
    Sets the value of the specified option.

    Parameters
    ----------
    key : str
        The key which should match a single option.
    value : object
        New value of option.

    Returns
    -------
    None
    """
    _check_option(key)
    _options_dict[key].validate(value)
    _options_values[key] = value


def reset_option(key: str) -> None:
    """
    Reset one option to their default value.

    Pass "all" as argument to reset all options.

    Parameters
    ----------
    key : str
        If specified only option will be reset.

    Returns
    -------
    None
    """
    _check_option(key)
    _options_values[key] = _options_dict[key].default


def _check_option(key: str) -> None:
    if key not in _options_dict:
        raise DDSException(
            "No such option: '{}'. Available options are [{}]".format(
                key, ", ".join(list(_options_dict.keys()))
            ),
            DDSErrorCode.UNKNOWN_OPTION,
        )
