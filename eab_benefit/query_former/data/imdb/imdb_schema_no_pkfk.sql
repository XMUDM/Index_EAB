--
-- PostgreSQL database dump
--

-- Dumped from database version 14.4 (Ubuntu 14.4-1.pgdg18.04+1)
-- Dumped by pg_dump version 14.4 (Ubuntu 14.4-1.pgdg18.04+1)

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: hypopg; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS hypopg WITH SCHEMA public;


--
-- Name: EXTENSION hypopg; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION hypopg IS 'Hypothetical advisor for PostgreSQL';


--
-- Name: tsm_system_rows; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS tsm_system_rows WITH SCHEMA public;


--
-- Name: EXTENSION tsm_system_rows; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION tsm_system_rows IS 'TABLESAMPLE method which accepts number of rows as a limit';


SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: aka_name; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.aka_name (
    id integer NOT NULL,
    person_id integer NOT NULL,
    name character varying,
    imdb_index character varying(3),
    name_pcode_cf character varying(11),
    name_pcode_nf character varying(11),
    surname_pcode character varying(11),
    md5sum character varying(65)
);


--
-- Name: aka_title; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.aka_title (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    title character varying,
    imdb_index character varying(4),
    kind_id integer NOT NULL,
    production_year integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    note character varying(72),
    md5sum character varying(32)
);


--
-- Name: cast_info; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.cast_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    movie_id integer NOT NULL,
    person_role_id integer,
    note character varying,
    nr_order integer,
    role_id integer NOT NULL
);


--
-- Name: char_name; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.char_name (
    id integer NOT NULL,
    name character varying NOT NULL,
    imdb_index character varying(2),
    imdb_id integer,
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);


--
-- Name: comp_cast_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.comp_cast_type (
    id integer NOT NULL,
    kind character varying(32) NOT NULL
);


--
-- Name: company_name; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.company_name (
    id integer NOT NULL,
    name character varying NOT NULL,
    country_code character varying(6),
    imdb_id integer,
    name_pcode_nf character varying(5),
    name_pcode_sf character varying(5),
    md5sum character varying(32)
);


--
-- Name: company_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.company_type (
    id integer NOT NULL,
    kind character varying(32)
);


--
-- Name: complete_cast; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.complete_cast (
    id integer NOT NULL,
    movie_id integer,
    subject_id integer NOT NULL,
    status_id integer NOT NULL
);


--
-- Name: info_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.info_type (
    id integer NOT NULL,
    info character varying(32) NOT NULL
);


--
-- Name: keyword; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.keyword (
    id integer NOT NULL,
    keyword character varying NOT NULL,
    phonetic_code character varying(5)
);


--
-- Name: kind_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.kind_type (
    id integer NOT NULL,
    kind character varying(15)
);


--
-- Name: link_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.link_type (
    id integer NOT NULL,
    link character varying(32) NOT NULL
);


--
-- Name: movie_companies; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.movie_companies (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    company_id integer NOT NULL,
    company_type_id integer NOT NULL,
    note character varying
);


--
-- Name: movie_info; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.movie_info (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info character varying NOT NULL,
    note character varying
);


--
-- Name: movie_info_idx; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.movie_info_idx (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info character varying NOT NULL,
    note character varying(1)
);


--
-- Name: movie_keyword; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.movie_keyword (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    keyword_id integer NOT NULL
);


--
-- Name: movie_link; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.movie_link (
    id integer NOT NULL,
    movie_id integer NOT NULL,
    linked_movie_id integer NOT NULL,
    link_type_id integer NOT NULL
);


--
-- Name: name; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.name (
    id integer NOT NULL,
    name character varying NOT NULL,
    imdb_index character varying(9),
    imdb_id integer,
    gender character varying(1),
    name_pcode_cf character varying(5),
    name_pcode_nf character varying(5),
    surname_pcode character varying(5),
    md5sum character varying(32)
);


--
-- Name: person_info; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.person_info (
    id integer NOT NULL,
    person_id integer NOT NULL,
    info_type_id integer NOT NULL,
    info character varying NOT NULL,
    note character varying
);


--
-- Name: role_type; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.role_type (
    id integer NOT NULL,
    role character varying(32) NOT NULL
);


--
-- Name: title; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE public.title (
    id integer NOT NULL,
    title character varying NOT NULL,
    imdb_index character varying(5),
    kind_id integer NOT NULL,
    production_year integer,
    imdb_id integer,
    phonetic_code character varying(5),
    episode_of_id integer,
    season_nr integer,
    episode_nr integer,
    series_years character varying(49),
    md5sum character varying(32)
);

--
-- PostgreSQL database dump complete
--

