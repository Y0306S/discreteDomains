/*+
NestLoop(title1 kind_type1)
NestLoop(title1 movie_info2 kind_type1)
NestLoop(title1 movie_info1 movie_info2 kind_type1)
NestLoop(cast_info1 kind_type1 title1 movie_info1 movie_info2)
NestLoop(cast_info1 kind_type1 title1 movie_info1 role_type1 movie_info2)
NestLoop(cast_info1 kind_type1 info_type1 title1 movie_info1 role_type1 movie_info2)
NestLoop(info_type1 role_type1 movie_info2 movie_keyword1 cast_info1 kind_type1 title1 movie_info1)
NestLoop(info_type1 role_type1 movie_info2 movie_keyword1 cast_info1 kind_type1 title1 movie_info1 info_type2)
NestLoop(info_type1 role_type1 movie_info2 movie_keyword1 cast_info1 kind_type1 title1 movie_info1 name1 info_type2)
NestLoop(info_type1 role_type1 movie_info2 movie_keyword1 cast_info1 kind_type1 title1 movie_info1 name1 keyword1 info_type2)
*/
SELECT
  COUNT(*)
FROM (((movie_keyword AS movie_keyword1 CROSS JOIN (((((movie_info AS movie_info2 CROSS JOIN (kind_type AS kind_type1 CROSS JOIN title AS title1)) CROSS JOIN movie_info AS movie_info1) CROSS JOIN cast_info AS cast_info1) CROSS JOIN role_type AS role_type1) CROSS JOIN info_type AS info_type1)) CROSS JOIN info_type AS info_type2) CROSS JOIN name AS name1) CROSS JOIN keyword AS keyword1
WHERE
  title1.id = cast_info1.movie_id
  AND title1.id = movie_info1.movie_id
  AND title1.id = movie_info2.movie_id
  AND title1.id = movie_keyword1.movie_id
  AND keyword1.id = movie_keyword1.keyword_id
  AND movie_info1.movie_id = movie_info2.movie_id
  AND movie_info1.info_type_id = info_type1.id
  AND movie_info2.info_type_id = info_type2.id
  AND (
    info_type1.id IN ('4')
  )
  AND (
    info_type2.id IN ('7')
  )
  AND title1.kind_id = kind_type1.id
  AND cast_info1.person_id = name1.id
  AND cast_info1.role_id = role_type1.id
  AND (
    movie_info1.info IN ('Czech', 'English', 'Hungarian', 'Italian', 'Japanese', 'Mandarin', 'Portuguese', 'Russian', 'Swedish')
  )
  AND (
    movie_info2.info IN ('OFM:35 mm', 'PCS:Kinescope', 'PCS:Spherical', 'PFM:35 mm', 'RAT:1.20 : 1', 'RAT:1.37 : 1', 'RAT:1.66 : 1', 'RAT:2.35 : 1')
  )
  AND (
    kind_type1.kind IN ('movie', 'tv movie')
  )
  AND (
    role_type1.role IN ('miscellaneous crew', 'writer')
  )
  AND (
    name1.gender IN ('f') OR name1.gender IS NULL
  )
  AND (
    title1.production_year <= 1975
  )
  AND (
    title1.production_year >= 1925
  )