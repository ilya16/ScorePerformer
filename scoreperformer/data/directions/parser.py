""" MusicXML performance direction data parsing. """

from musicxml_parser.playable_notes import get_playable_notes

from .articulation import ARTICULATION_PREFIX
from .dynamic import DYNAMIC_PREFIX, ABS_DYNAMIC_KEYS, REL_DYNAMIC_KEYS, hairpin_word_regularization
from .tempo import TEMPO_PREFIX, TEMPO_KEYS
from .words import extract_direction_by_keys, word_regularization


def get_part_directions(part):
    directions = []
    for measure_idx, measure in enumerate(part.measures):
        for direction in measure.directions:
            direction.type['measure'] = measure_idx
            directions.append(direction)

    directions.sort(key=lambda x: x.xml_position)
    cleaned_direction = []
    for i, d in enumerate(directions):
        if d.type is None:
            continue

        if d.type['type'] == 'none':
            for j in range(i):
                prev_dir = directions[i - j - 1]
                if 'number' in prev_dir.type.keys():
                    prev_key = prev_dir.type['type']
                    prev_num = prev_dir.type['number']
                else:
                    continue
                if prev_num == d.type['number']:
                    if prev_key == "crescendo":
                        d.type['type'] = 'crescendo'
                        break
                    elif prev_key == "diminuendo":
                        d.type['type'] = 'diminuendo'
                        break
        cleaned_direction.append(d)

    return cleaned_direction


def get_directions(doc):
    return [get_part_directions(part) for part in doc.parts]


def parse_directions(doc, score_directions=None, delete_unmatched=False, delete_duplicates=False, ticks_scale=1.):
    score_directions_init = get_directions(doc) if score_directions is None else score_directions

    last_note = doc.parts[-1].measures[-1].notes[-1].note_duration
    max_xml_position = max(doc._state.xml_position, last_note.xml_position + last_note.duration)

    # anacrusis
    measure_pos = doc.get_measure_positions()
    xml_shift = max(0, measure_pos[2] - 2 * measure_pos[1] + measure_pos[0])

    score_directions = []
    for part_idx, part_directions_init in enumerate(score_directions_init):
        active_dynamic = None
        active_tempo = None
        active_hairpins = {}
        part_directions = []
        for d_idx, d in enumerate(part_directions_init):
            d_data, d_dict = d.type, None
            if d_data['type'] == 'dynamic':
                d_dict = {
                    'type': d_data['type'],
                    'start': d.xml_position,
                    'end': max_xml_position
                }
                abs_dynamic = extract_direction_by_keys(d_data['content'], ABS_DYNAMIC_KEYS)
                rel_dynamic = extract_direction_by_keys(d_data['content'], REL_DYNAMIC_KEYS)

                if abs_dynamic is not None:
                    d_dict['type'] += '/' + abs_dynamic
                    if active_dynamic is not None:
                        active_dynamic['end'] = d.xml_position
                    active_dynamic = d_dict
                elif rel_dynamic is not None:
                    d_dict['type'] += '/' + rel_dynamic
                    d_dict['end'] = d_dict['start']
                else:
                    continue
            elif d_data['type'] in ('crescendo', 'diminuendo'):
                key = f'{d_data["type"]}_{d_data["number"]}'
                if d_data['content'] == 'start':
                    active_hairpins[key] = d
                elif d_data['content'] == 'stop':
                    start_d = active_hairpins.pop(key, None)
                    if not start_d:
                        continue
                    d_dict = {
                        'type': 'dynamic' + '/' + d_data['type'],
                        'start': start_d.xml_position,
                        'end': d.xml_position
                    }
            elif d_data['type'] == 'words':
                word = word_regularization(d_data['content'])
                word = hairpin_word_regularization(word)
                tempo_word = extract_direction_by_keys(word, TEMPO_KEYS)

                if word in ('crescendo', 'diminuendo'):
                    d_dict = {'type': DYNAMIC_PREFIX}
                elif tempo_word is not None:
                    word = tempo_word
                    d_dict = {'type': TEMPO_PREFIX}
                    if active_tempo is not None:
                        active_tempo['end'] = d.xml_position
                    active_tempo = d_dict
                elif delete_unmatched:
                    continue
                else:
                    d_dict = {'type': d_data['type']}

                d_dict['type'] += '/' + word
                d_dict.update(**{
                    'start': d.xml_position,
                    'end': max_xml_position if d_dict['type'] == 'tempo' else d.xml_position
                })
            else:
                d_dict = None

            if d_dict is not None:
                d_dict.update(**{
                    'part': part_idx,
                    'staff': int(d.staff) if d.staff is not None else 1
                })
                part_directions.append(d_dict)

        # parse note articulations
        def _build_note_articulation_dict(note, content):
            return {
                'type': ARTICULATION_PREFIX + '/' + content,
                'start': note.note_duration.xml_position,
                'end': note.note_duration.xml_position + note.note_duration.duration,
                'pitch': note.pitch[1],
                'part': part_idx,
                'staff': int(note.staff) if note.staff is not None else 1
            }

        part_notes, _ = get_playable_notes(doc.parts[part_idx])
        for note in part_notes:
            if note.note_notations.is_arpeggiate:
                part_directions.append(_build_note_articulation_dict(note, 'arpeggiate'))
            if note.note_notations.is_fermata:
                part_directions.append(_build_note_articulation_dict(note, 'fermata'))
            if note.note_notations.is_staccato:
                part_directions.append(_build_note_articulation_dict(note, 'staccato'))
            if note.note_notations.is_tenuto:
                part_directions.append(_build_note_articulation_dict(note, 'tenuto'))

        # scale xml positions if needed
        if xml_shift != 0 or ticks_scale != 1.:
            for d_dict in part_directions:
                d_dict['start'] = int(ticks_scale * (d_dict['start'] + xml_shift))
                d_dict['end'] = int(ticks_scale * (d_dict['end'] + xml_shift))

        # sort directions
        part_directions = list(sorted(part_directions, key=lambda d: (d['start'], d['type'], d['end'])))

        if delete_duplicates:
            i = 0
            while i < len(part_directions) - 1:
                d_dict, next_d_dict = part_directions[i], part_directions[i + 1]
                if d_dict['type'] == next_d_dict['type'] and d_dict['start'] == next_d_dict['start']:
                    del part_directions[i + 1]
                    continue
                i += 1

        score_directions.append(part_directions)

    return score_directions
