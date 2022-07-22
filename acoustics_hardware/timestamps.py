import datetime
import re
import os

# TODO: make a timestamper class instead of functions?


def parse_timestamp(ts):
    sep = r'[-,_,:,T, ]?'

    year = r'(?P<year>\d{4})'
    month = r'(?P<month>\d{2})'
    day = r'(?P<day>\d{2})'
    hour = r'(?P<hour>\d{2})?'
    minute = r'(?P<minute>\d{2})?'
    second = r'(?P<second>\d{2})?'
    fraction = r'\.?(?P<fraction>\d+)?'

    pattern = sep.join([year, month, day, hour, minute, second]) + fraction
    regex = re.compile(pattern)
    match = regex.search(ts)
    if match is None:
        raise ValueError(f'Count not parse timestamp {ts}')

    #     raise ValueError(f'Count not parse timestamp {ts}')

    args = []
    for arg in match.groups():
        if arg is not None:
            args.append(int(arg))
        else:
            break
    return datetime.datetime(*args)
    # TODO: convert the matched strings to ints
    # TODO: fill two digit year to four digit year
    # TODO: raise something meaningful on missing info
    return datetime.datetime(
        int(groups['year']),
        int(groups['month']),
        int(groups['day']),
        int(groups['hour']),
        int(groups['minute']),
        int(groups['second']),
        int(groups['fraction']),
    )


def timestamp(format='ISO', microseconds=False, utc=False):
    now = datetime.datetime.now(datetime.timezone.utc)
    if not utc:
        now = now.astimezone()
    if format.lower() == 'iso':
        return now.isoformat(timespec='microseconds' if microseconds else 'seconds')
    if format.lower() == 'filename':
        return now.strftime('%Y-%m-%d_%H-%M-%S' + ('.%f' if microseconds else ''))
    if format.lower() == 'epoch':
        return now.timestamp() if microseconds else int(now.timestamp())
    raise ValueError(f"Unknown timestamp format {format}")


def newest_timestamed_filename(directory, n_files=1):
    return_files = []
    files = os.listdir(directory)
    for _ in range(n_files):
        newest = None
        for file in files:
            try:
                stamp = parse_timestamp(file)
            except ValueError:
                continue
            if newest is None:
                newest = (file, stamp)
            elif stamp > newest[1]:
                newest = (file, stamp)
        if newest is not None:
            return_files.append(newest[0])
            files.remove(file)
    if len(return_files) < n_files:
        raise ValueError('No matching files found')
    if n_files == 1:
        return return_files[0]
    return return_files
